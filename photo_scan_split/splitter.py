"""Detect and split individual photos from a scanned image.

Uses watershed segmentation with distance transform to reliably separate
photos placed with visible gaps on the scanner bed.
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
    open_k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_k, iterations=2)
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k, iterations=2)

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
        sure_fg[(blob_dist > 0.5 * local_max)] = 255

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
