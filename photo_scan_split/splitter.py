"""Detect and split individual photos from a scanned image.

Uses watershed segmentation with distance transform to reliably separate
photos placed on the scanner bed.  Includes face-detection-based auto
orientation for extracted photos.
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


def _auto_orient(pil_img: Image.Image) -> Image.Image:
    """Correct photo orientation using face detection.

    Tries all four 90-degree rotations, runs frontal and profile face
    detection on each, and picks the rotation with the strongest
    detection.  Returns the original unchanged when no faces are found.
    """
    img = np.array(pil_img)
    if img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY)
    elif img.ndim == 2:
        gray = img
    else:
        return pil_img

    max_dim = 800
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    )
    profile_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_profileface.xml"
    )

    candidates = [
        (0, small),
        (90, cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE)),
        (180, cv2.rotate(small, cv2.ROTATE_180)),
        (270, cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE)),
    ]

    best_angle = 0
    best_score = 0

    for angle, rotated in candidates:
        min_face = max(30, int(min(rotated.shape[:2]) * 0.04))
        detect_kw = dict(
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_face, min_face),
        )
        frontal = face_cascade.detectMultiScale(rotated, **detect_kw)
        profile = profile_cascade.detectMultiScale(rotated, **detect_kw)

        score = len(frontal) * 2 + len(profile)
        if score > best_score:
            best_score = score
            best_angle = angle

    if best_angle == 0 or best_score == 0:
        return pil_img

    transpose_map = {
        90: Image.Transpose.ROTATE_90,
        180: Image.Transpose.ROTATE_180,
        270: Image.Transpose.ROTATE_270,
    }
    return pil_img.transpose(transpose_map[best_angle])


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
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
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

        pil_img = _auto_orient(pil_img)

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
