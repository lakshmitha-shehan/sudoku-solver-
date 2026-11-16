"""Sudoku screenshot -> 450x450 aligned grid -> 81 cells and ONE occupancy decision.

This module owns the blank/digit/uncertain classification. Nothing runs at import.
Requires: opencv-python, numpy. Supports mostly straight Sudoku screenshots;
strongly tilted photographs still need a better grid detector.
"""
from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np

DEFAULT_IMAGE = Path(__file__).resolve().parent / "images" / "13.png"
SIDE = 450
CELL = SIDE // 9


@dataclass
class SudokuImage:
    original: np.ndarray
    warped: np.ndarray
    boundary_preview: np.ndarray
    cells: list
    cell_status: list      # '0' = empty, 'D' = real foreground, '?' = uncertain
    digit_masks: list      # selected white-on-black digit mask or None
    mask_variants: list    # candidate masks, generated ONLY here
    corners: np.ndarray
    method: str


def _line_map(binary):
    h, w = binary.shape
    k = max(12, min(h, w) // 19)
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                   cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                   cv2.getStructuringElement(cv2.MORPH_RECT, (1, k)))
    return cv2.bitwise_or(horizontal, vertical)


def _order_four(points):
    pts = np.asarray(points, np.float32).reshape(4, 2)
    # Ordering by x after splitting into top/bottom halves avoids duplicate
    # indices that can occur with sum/difference on nearly diamond shapes.
    top_bottom = pts[np.argsort(pts[:, 1])]
    top = top_bottom[:2][np.argsort(top_bottom[:2, 0])]
    bottom = top_bottom[2:][np.argsort(top_bottom[2:, 0])]
    return np.float32([top[0], top[1], bottom[1], bottom[0]])


def _grid_fit_axis(profile):
    """Find ten approximately equally spaced grid borders along one dimension."""
    p = np.asarray(profile, np.float32)
    length = len(p)
    # Line thickness varies; use local maximum within +-2 pixels.
    p = np.array([p[max(0,i-2):min(length,i+3)].max()
                  for i in range(length)], dtype=np.float32)
    best = (-1.0, None)
    weights = np.float32([.7] + [1.0]*8 + [.7])
    for spacing in np.arange(length*.72/9, length*1.04/9, .5):
        max_start = int(length - 1 - 9*spacing)
        if max_start < 0:
            continue
        starts = np.arange(max_start+1, dtype=np.float32)
        indices = np.rint(starts[:,None] +
                 np.arange(10, dtype=np.float32)[None,:]*spacing).astype(int)
        indices = np.clip(indices,0,length-1)
        scores = (p[indices]*weights).sum(axis=1)/weights.sum()
        index = int(np.argmax(scores))
        if scores[index] > best[0]:
            best = (float(scores[index]), (float(starts[index]),float(spacing)))
    return best


def _detect_regular_grid(image):
    """Locate real grid bounds even when corners are rounded or outer margins exist."""
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,35,100)
    k = max(12,min(gray.shape)//22)
    vert = cv2.morphologyEx(edges,cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT,(1,k)))
    horiz = cv2.morphologyEx(edges,cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT,(k,1)))
    score_x, fit_x = _grid_fit_axis(np.sum(vert>0, axis=0))
    score_y, fit_y = _grid_fit_axis(np.sum(horiz>0, axis=1))
    h,w = gray.shape
    if fit_x is None or fit_y is None or score_x < .16*h or score_y < .16*w:
        return None
    x0,dx = fit_x
    y0,dy = fit_y
    if not (.89 < dx/dy < 1.12):
        return None
    x1 = x0 + 9*dx
    y1 = y0 + 9*dy
    if x1 >= w+2 or y1 >= h+2:
        return None
    return np.float32([[x0,y0],[x1,y0],[x1,y1],[x0,y1]])


def _find_board(image):
    regular = _detect_regular_grid(image)
    if regular is not None:
        return regular, 'regular line spacing'
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape
    masks = []
    for src in (blur, cv2.bitwise_not(blur)):
        binary = cv2.adaptiveThreshold(src, 255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV, 11, 2)
        lines = _line_map(binary)
        closed = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        masks.append((lines, closed))

    candidates = []
    image_area = h * w
    # Try BOTH polarities, not only a median-brightness guess.
    for variant, (lines, closed) in enumerate(masks):
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:15]:
            area = cv2.contourArea(contour)
            if area < 0.22 * image_area:
                continue
            perimeter = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, .022 * perimeter, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue
            pts = _order_four(approx)
            top = np.linalg.norm(pts[1] - pts[0])
            bottom = np.linalg.norm(pts[2] - pts[3])
            left = np.linalg.norm(pts[3] - pts[0])
            right = np.linalg.norm(pts[2] - pts[1])
            aspect = (top + bottom) / max(1, left + right)
            if not (.72 < aspect < 1.38):
                continue
            # Prefer large, almost square regions; straight screenshot puzzle
            # typically occupies most of the available image.
            score = area / image_area - .28 * abs(np.log(aspect))
            candidates.append((score, pts, f'contour polarity {variant}'))

        # Backup: bounds of detected long grid strokes.
        nonzero = cv2.findNonZero(lines)
        if nonzero is not None:
            x, y, bw, bh = cv2.boundingRect(nonzero)
            ar = bw / max(1, bh)
            if (bw >= .55 * w and bh >= .55 * h and .74 < ar < 1.34):
                pts = np.float32([[x, y], [x + bw - 1, y],
                                  [x + bw - 1, y + bh - 1], [x, y + bh - 1]])
                score = .72 * (bw * bh / image_area) - .2 * abs(np.log(ar))
                candidates.append((score, pts, f'line bounds polarity {variant}'))
    if not candidates:
        raise ValueError('Cannot find Sudoku boundary; use a tightly cropped screenshot.')
    _, corners, method = max(candidates, key=lambda item: item[0])
    return corners, method


def _extract_foreground(cell):
    """Classify a cell ONCE and provide digit masks for the OCR module.

    0 means demonstrably blank; D means plausible central glyph; ? means ambiguous.
    The 8px inset discards frame lines and rounded-corner debris; unlike the old
    5+3px cropping it is one controlled calculation from the original 50px cell.
    """
    crop = cell[8:-8, 8:-8].copy()    # 34 x 34 pixels
    h, w = crop.shape[:2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Use central cell background; avoid digit pixels by sampling small corners.
    corners = np.concatenate([
        crop[:6, :6].reshape(-1, 3), crop[:6, -6:].reshape(-1, 3),
        crop[-6:, :6].reshape(-1, 3), crop[-6:, -6:].reshape(-1, 3)
    ]).astype(np.float32)
    bg = np.median(corners, axis=0)
    diff = np.max(np.abs(crop.astype(np.float32) - bg), axis=2)
    contrast = float(np.percentile(diff, 94))
    proposals = [(f'color_{t}', (diff >= t).astype(np.uint8) * 255)
                 for t in (10, 17, 26, 38)]
    if np.percentile(gray, 97) - np.percentile(gray, 3) >= 13:
        _, otsu = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if cv2.countNonZero(otsu) > otsu.size / 2:
            otsu = cv2.bitwise_not(otsu)
        proposals.append(('otsu', otsu))

    variants = []
    for name, raw in proposals:
        mask = raw.copy()
        # Cut 1px of the new crop border; rejected pieces must not extend to edge.
        mask[0, :] = 0; mask[-1, :] = 0
        mask[:, 0] = 0; mask[:, -1] = 0
        n, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        candidates = []
        for i in range(1, n):
            x, y, cw, ch, area = map(int, stats[i])
            cx, cy = centroids[i]
            if not (area >= 7 and ch >= 7 and cw >= 1 and
                    cw <= .80*w and ch <= h and
                    .17*w < cx < .83*w and .15*h < cy < .85*h):
                continue
            # A line from a cell border is not an interior digit.
            if x <= 1 or x + cw >= w-1:
                continue
            candidates.append(i)
        if not candidates:
            continue
        # Keep the most central, meaningful digit component.
        primary = max(candidates, key=lambda i:
            float(stats[i, cv2.CC_STAT_AREA]) /
            (1.0 + 0.8 * (abs(centroids[i][0]-w/2) / w +
                          abs(centroids[i][1]-h/2) / h)))
        x, y, cw, ch, area = map(int, stats[primary])
        clean = np.zeros_like(mask)
        for i in candidates:
            xx, yy, ww, hh, _ = map(int, stats[i])
            if (xx <= x + cw + 2 and xx + ww >= x-2 and
                yy <= y + ch + 2 and yy + hh >= y-2):
                clean[labels == i] = 255
        variants.append((name, clean, area))

    if variants:
        variants.sort(key=lambda t: (t[0].startswith('color_'), t[2]),
                      reverse=True)
        return 'D', variants[0][1], [(name, m) for name, m, _ in variants]

    # All masks were evaluated. Uniform/noisy-free center => empty.
    # If a significant amount of central ink remains but cannot be isolated,
    # flag rather than silently replacing a faint given with zero.
    central = diff[7:27, 7:27]
    active = int(np.count_nonzero(central > 22))
    if contrast > 35 and 15 <= active <= 180:
        return '?', None, []
    return '0', None, []


def process_sudoku(image_path=DEFAULT_IMAGE, debug=False):
    image_path = Path(image_path)
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f'Image not found: {image_path}')
    corners, method = _find_board(image)
    destination = np.float32([[0, 0], [SIDE-1, 0],
                              [SIDE-1, SIDE-1], [0, SIDE-1]])
    matrix = cv2.getPerspectiveTransform(corners, destination)
    warped = cv2.warpPerspective(image, matrix, (SIDE, SIDE))
    boundary = image.copy()
    cv2.polylines(boundary, [np.int32(corners)], True, (0,255,0), 2)

    cells, status, masks, variants = [], [], [], []
    for row in range(9):
        cell_row, stat_row, mask_row, options_row = [], [], [], []
        for col in range(9):
            cell = warped[row*CELL:(row+1)*CELL,
                          col*CELL:(col+1)*CELL].copy()
            state, mask, options = _extract_foreground(cell)
            cell_row.append(cell)
            stat_row.append(state)
            mask_row.append(mask)
            options_row.append(options)
        cells.append(cell_row)
        status.append(stat_row)
        masks.append(mask_row)
        variants.append(options_row)

    result = SudokuImage(image, warped, boundary, cells, status, masks,
                         variants, corners, method)
    if debug:
        print('Grid detection:', method)
        print('Four corners:', corners.astype(int).tolist())
        print('Cell occupancy (0 blank, D digit, ? uncertain):')
        for row in status:
            print(' '.join(row))
    return result


def draw_status(result):
    preview = np.zeros_like(result.warped)
    for r in range(9):
        for c in range(9):
            state = result.cell_status[r][c]
            color = (0,255,0) if state == 'D' else ((0,0,255) if state == '?' else (130,130,130))
            cv2.putText(preview, state, (c*CELL+16, r*CELL+35),
                        cv2.FONT_HERSHEY_SIMPLEX, .8, color, 2)
    for i in range(10):
        thick = 2 if i % 3 == 0 else 1
        cv2.line(preview, (i*CELL,0), (i*CELL,SIDE-1), (0,155,0), thick)
        cv2.line(preview, (0,i*CELL), (SIDE-1,i*CELL), (0,155,0), thick)
    return preview


if __name__ == '__main__':
    result = process_sudoku(DEFAULT_IMAGE, debug=True)
    cv2.imshow('Original', result.original)
    cv2.imshow('Boundary', result.boundary_preview)
    cv2.imshow('Straightened Sudoku', result.warped)
    cv2.imshow('Cell Occupancy', draw_status(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
