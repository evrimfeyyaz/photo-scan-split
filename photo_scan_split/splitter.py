"""Detect and split individual photos from a scanned image.

Uses watershed segmentation with distance transform and edge-based gap
detection to reliably separate photos, including ones placed close
together or touching on the scanner bed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass
class DetectedPhoto:
    """A single photo region detected within a scan."""

    image: Image.Image
    bbox: tuple[int, int, int, int]  # x, y, w, h in the original scan
    index: int


def _remove_contained(photos: list[DetectedPhoto]) -> list[DetectedPhoto]:
    """Drop photos whose centre falls inside a larger photo's bbox."""
    keep: list[DetectedPhoto] = []
    for i, p in enumerate(photos):
        px, py, pw, ph = p.bbox
        cx, cy = px + pw // 2, py + ph // 2
        p_area = pw * ph

        contained = False
        for j, other in enumerate(photos):
            if i == j:
                continue
            ox, oy, ow, oh = other.bbox
            if ow * oh > p_area and ox <= cx <= ox + ow and oy <= cy <= oy + oh:
                contained = True
                break
        if not contained:
            keep.append(p)
    return keep


def _find_peak_positions(profile: np.ndarray, min_distance: int) -> list[int]:
    """Find significant peaks in a 1D profile (above median + 2*std).

    Peaks closer than *min_distance* apart are merged, keeping the
    strongest.  Returns a list of peak indices into *profile*.
    """
    if len(profile) == 0 or profile.max() == 0:
        return []
    med = float(np.median(profile))
    std = float(np.std(profile))
    if std < 1e-6:
        return []
    threshold = med + 2.0 * std

    above = profile > threshold
    peaks: list[int] = []
    i = 0
    while i < len(profile):
        if above[i]:
            j = i
            while j < len(profile) and above[j]:
                j += 1
            peak = i + int(np.argmax(profile[i:j]))
            if not peaks or (peak - peaks[-1]) >= min_distance:
                peaks.append(peak)
            elif profile[peak] > profile[peaks[-1]]:
                peaks[-1] = peak
            i = j
        else:
            i += 1
    return peaks


def _cut_touching_blobs(
    mask: np.ndarray,
    gray: np.ndarray,
    min_area: int,
) -> np.ndarray:
    """Detect and sever boundaries between touching or very close photos.

    For each connected component large enough to contain multiple photos,
    projects directional Sobel gradient magnitudes onto vertical and
    horizontal axes.  A sharp peak in gradient density indicates a
    content boundary between two photos — a thin cut is drawn through the
    mask at that position so that downstream watershed can separate them.
    """
    h_img, w_img = mask.shape
    num_blobs, blob_labels = cv2.connectedComponents(mask)
    if num_blobs <= 1:
        return mask

    min_photo_dim = max(int(min(h_img, w_img) * 0.08), 50)
    cut_half = 2  # total cut width = 2 * cut_half + 1 = 5 px

    grad_x = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3))
    grad_y = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3))

    result = mask.copy()

    for blob_id in range(1, num_blobs):
        blob_mask = blob_labels == blob_id
        blob_area = int(np.sum(blob_mask))
        if blob_area < min_area * 1.5:
            continue

        ys, xs = np.where(blob_mask)
        y0, y1 = int(ys.min()), int(ys.max())
        x0, x1 = int(xs.min()), int(xs.max())
        bh, bw = y1 - y0 + 1, x1 - x0 + 1

        crop_blob = blob_mask[y0 : y1 + 1, x0 : x1 + 1]

        # --- vertical cuts (side-by-side photos) ---
        if bw >= min_photo_dim * 1.5:
            crop_gx = grad_x[y0 : y1 + 1, x0 : x1 + 1].copy()
            crop_gx[~crop_blob] = 0.0

            col_grad = np.sum(crop_gx, axis=0)
            col_cov = np.maximum(np.sum(crop_blob, axis=0).astype(np.float64), 1.0)
            density = col_grad / col_cov

            k = max(5, bw // 30)
            if k % 2 == 0:
                k += 1
            smooth = cv2.GaussianBlur(
                density.reshape(1, -1).astype(np.float32), (1, k), 0,
            ).flatten()

            margin = max(min_photo_dim // 2, bw // 8)
            if margin < bw - margin:
                interior = smooth[margin : bw - margin]
                for p in _find_peak_positions(interior, min_photo_dim):
                    col_abs = p + margin + x0
                    for dc in range(-cut_half, cut_half + 1):
                        cc = col_abs + dc
                        if 0 <= cc < w_img:
                            result[y0 : y1 + 1, cc][blob_mask[y0 : y1 + 1, cc]] = 0

        # --- horizontal cuts (top-bottom photos) ---
        if bh >= min_photo_dim * 1.5:
            crop_gy = grad_y[y0 : y1 + 1, x0 : x1 + 1].copy()
            crop_gy[~crop_blob] = 0.0

            row_grad = np.sum(crop_gy, axis=1)
            row_cov = np.maximum(np.sum(crop_blob, axis=1).astype(np.float64), 1.0)
            density = row_grad / row_cov

            k = max(5, bh // 30)
            if k % 2 == 0:
                k += 1
            smooth = cv2.GaussianBlur(
                density.reshape(-1, 1).astype(np.float32), (k, 1), 0,
            ).flatten()

            margin = max(min_photo_dim // 2, bh // 8)
            if margin < bh - margin:
                interior = smooth[margin : bh - margin]
                for p in _find_peak_positions(interior, min_photo_dim):
                    row_abs = p + margin + y0
                    for dr in range(-cut_half, cut_half + 1):
                        rr = row_abs + dr
                        if 0 <= rr < h_img:
                            result[rr, x0 : x1 + 1][blob_mask[rr, x0 : x1 + 1]] = 0

    return result


def split_photos(
    image: Image.Image,
    min_area_pct: float = 3.0,
    bg_threshold: int = 230,
) -> list[DetectedPhoto]:
    """Detect and extract individual photos from a scanned image.

    Photos should be placed with some visible gap between them on the
    scanner bed (white background visible between photos).

    Args:
        image: The full scanned image as a PIL Image.
        min_area_pct: Minimum photo area as a percentage of the total scan
            area.  Regions smaller than this are discarded as noise.
        bg_threshold: Grayscale threshold to separate photos from the white
            scanner background.  Pixels brighter than this are background.

    Returns:
        A list of DetectedPhoto objects, sorted left-to-right then
        top-to-bottom.
    """
    img_array = np.array(image)
    if img_array.ndim == 2:
        gray = img_array
        color = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
    else:
        gray = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2GRAY)
        color = cv2.cvtColor(img_array[:, :, :3], cv2.COLOR_RGB2BGR)

    h_img, w_img = gray.shape
    total_area = h_img * w_img
    min_area = total_area * (min_area_pct / 100.0)

    # --- Binary mask: photos = white, background = black ---
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, bg_threshold, 255, cv2.THRESH_BINARY_INV)

    # Open removes noise; close bridges small bright gaps within photos.
    # The close kernel is kept small to avoid merging nearby photos.
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=2)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=1)

    # --- Sever touching / very close photos ---
    mask = _cut_touching_blobs(mask, gray, int(min_area))

    # --- Sure background ---
    bg_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    sure_bg = cv2.dilate(mask, bg_kernel, iterations=3)

    # --- Sure foreground via per-blob adaptive distance transform ---
    num_blobs, blob_labels = cv2.connectedComponents(mask)
    filled = np.zeros_like(mask)
    for blob_id in range(1, num_blobs):
        blob_pixels = np.uint8(blob_labels == blob_id) * 255
        cnts, _ = cv2.findContours(blob_pixels, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            cv2.drawContours(filled, cnts, -1, 255, cv2.FILLED)

    dist_transform = cv2.distanceTransform(filled, cv2.DIST_L2, 5)

    sure_fg = np.zeros_like(mask)
    for blob_id in range(1, num_blobs):
        blob_mask_b = blob_labels == blob_id
        blob_dist = dist_transform.copy()
        blob_dist[~blob_mask_b] = 0
        local_max = blob_dist.max()
        if local_max == 0:
            continue
        sure_fg[(blob_dist > 0.35 * local_max)] = 255

    if sure_fg.max() == 0:
        return []

    # --- Unknown region ---
    unknown = cv2.subtract(sure_bg, sure_fg)

    # --- Markers for watershed ---
    num_labels, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    # --- Watershed ---
    cv2.watershed(color, markers)

    # --- Extract each labeled region ---
    photos: list[DetectedPhoto] = []
    for label in range(2, num_labels + 1):
        region_mask = np.uint8((markers == label) & (mask > 0)) * 255
        contours, _ = cv2.findContours(region_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(contour) < min_area:
            continue

        # Trim edges: keep rows/cols with >15% mask coverage
        x, y, w, h = cv2.boundingRect(contour)
        mask_crop = region_mask[y : y + h, x : x + w]

        col_cov = np.mean(mask_crop > 0, axis=0)
        row_cov = np.mean(mask_crop > 0, axis=1)
        cc = np.where(col_cov > 0.15)[0]
        rc = np.where(row_cov > 0.15)[0]
        if len(cc) == 0 or len(rc) == 0:
            continue

        ml, mr = int(cc[0]), int(cc[-1]) + 1
        mt, mb = int(rc[0]), int(rc[-1]) + 1

        cropped = img_array[y + mt : y + mb, x + ml : x + mr]
        if cropped.size == 0:
            continue

        if cropped.ndim == 3 and cropped.shape[2] >= 3:
            pil_img = Image.fromarray(cropped[:, :, :3])
        elif cropped.ndim == 2:
            pil_img = Image.fromarray(cropped, mode="L")
        else:
            pil_img = Image.fromarray(cropped)

        photos.append(DetectedPhoto(
            image=pil_img,
            bbox=(x + ml, y + mt, mr - ml, mb - mt),
            index=0,
        ))

    photos = _remove_contained(photos)

    photos.sort(key=lambda p: (p.bbox[1] // 100, p.bbox[0]))
    for i, photo in enumerate(photos):
        photo.index = i

    return photos
