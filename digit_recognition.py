
"""Sudoku digit OCR, only for cells that image_processing marks as occupied.

This file NEVER classifies a cell as blank by running its own thresholding.
Requires: opencv-python, numpy, pytesseract, and the Tesseract Windows exe.
"""

from collections import defaultdict
from pathlib import Path
import argparse
import cv2
import numpy as np
import pytesseract

from image_processing import process_sudoku, DEFAULT_IMAGE, SIDE, CELL


TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
MIN_OCR_CONF = 53.0

# Only after visually checking the original image.
# Zero-based row and column.
# Example: MANUAL_CORRECTIONS = {(2, 7): 9}
# Do not apply corrections from one image to another.
MANUAL_CORRECTIONS = {}


def normalize_mask(mask, size=36, contents=28):
    """Normalize a digit for same-puzzle matching."""

    y, x = np.where(mask > 0)

    if not len(x):
        return None

    piece = mask[y.min():y.max()+1, x.min():x.max()+1]
    h, w = piece.shape

    scale = contents / max(h, w)

    rw = max(1, round(w * scale))
    rh = max(1, round(h * scale))

    small = cv2.resize(
        piece, (rw, rh),
        interpolation=cv2.INTER_AREA
    )

    output = np.zeros((size, size), np.uint8)

    output[
        (size-rh)//2:(size-rh)//2+rh,
        (size-rw)//2:(size-rw)//2+rw
    ] = (small > 110).astype(np.uint8)

    return output


def make_ocr_image(mask, height=62):
    """Prepare a white image with a centered black digit."""

    y, x = np.where(mask > 0)

    if not len(x):
        return None

    piece = mask[y.min():y.max()+1, x.min():x.max()+1]
    h, w = piece.shape

    scale = height / max(h, w)

    rw = max(1, round(w * scale))
    rh = max(1, round(h * scale))

    resized = cv2.resize(
        piece,
        (rw, rh),
        interpolation=cv2.INTER_CUBIC
    )

    page = np.full((96, 96), 255, np.uint8)
    ink = np.where(resized >= 128, 0, 255).astype(np.uint8)

    px = (96 - rw) // 2
    py = (96 - rh) // 2

    page[py:py+rh, px:px+rw] = ink

    return page


def tesseract_one(mask, psm=10, height=62):
    """Recognize one digit with Tesseract."""

    page = make_ocr_image(mask, height)

    if page is None:
        return None

    cfg = (
        f'--oem 1 --psm {psm} '
        '-c tessedit_char_whitelist=123456789 '
        '-c load_system_dawg=0 -c load_freq_dawg=0'
    )

    try:
        data = pytesseract.image_to_data(
            page,
            config=cfg,
            output_type=pytesseract.Output.DICT,
            timeout=3.0
        )

    except (RuntimeError, pytesseract.TesseractError):
        return None

    results = []

    for text, confidence in zip(data['text'], data['conf']):

        text = text.strip()

        if len(text) == 1 and text in '123456789':

            conf = float(confidence)

            if conf >= 0:
                results.append((int(text), conf))

    return max(results, key=lambda p: p[1]) if results else None


def similarity(a, b):
    """Compare two digit shapes allowing small translations."""

    if a is None or b is None:
        return 0.0

    a = (a > 0).astype(np.uint8)
    b = (b > 0).astype(np.uint8)

    best = 0.0

    for dy in (-2, -1, 0, 1, 2):
        for dx in (-2, -1, 0, 1, 2):

            shifted = np.zeros_like(b)

            sy0 = max(0, -dy)
            sy1 = min(len(b), len(b) - dy)

            sx0 = max(0, -dx)
            sx1 = min(len(b), len(b) - dx)

            if sy0 >= sy1 or sx0 >= sx1:
                continue

            shifted[
                sy0+dy:sy1+dy,
                sx0+dx:sx1+dx
            ] = b[sy0:sy1, sx0:sx1]

            union = np.count_nonzero(a | shifted)

            if union:
                score = np.count_nonzero(a & shifted) / union
                best = max(best, score)

    return best


def find_conflicts(board):
    """Find coordinates with duplicate Sudoku givens."""

    bad = set()
    groups = []

    # Rows
    for r in range(9):
        groups.append([(r, c) for c in range(9)])

    # Columns
    for c in range(9):
        groups.append([(r, c) for r in range(9)])

    # 3x3 boxes
    for br in range(0, 9, 3):
        for bc in range(0, 9, 3):

            groups.append([
                (r, c)
                for r in range(br, br+3)
                for c in range(bc, bc+3)
            ])

    for group in groups:

        seen = defaultdict(list)

        for r, c in group:

            n = board[r][c]

            if n not in (None, 0):
                seen[n].append((r, c))

        for same in seen.values():
            if len(same) > 1:
                bad.update(same)

    return bad


def legal_at(board, r, c, n):
    """Check if a candidate obeys Sudoku rules."""

    return (
        all(
            board[r][j] != n
            for j in range(9)
            if j != c
        )
        and
        all(
            board[i][c] != n
            for i in range(9)
            if i != r
        )
        and
        all(
            board[i][j] != n
            for i in range(r//3*3, r//3*3+3)
            for j in range(c//3*3, c//3*3+3)
            if (i, j) != (r, c)
        )
    )


def recognize(result):
    """Recognize Sudoku numbers.

    Returns:
        board: 0 = empty, 1..9 = digit, None = uncertain
        review: coordinates requiring review
        details: OCR confidence and matching information
    """

    details = {}

    # Step 1: Recognize each occupied cell
    for r in range(9):
        for c in range(9):

            state = result.cell_status[r][c]

            # Empty-cell detection belongs to image_processing.py
            if state == '0':

                details[r, c] = {
                    'blank': True,
                    'scores': {},
                    'raw': {},
                    'shape': None
                }

                continue

            variants = result.mask_variants[r][c]

            if not variants:

                details[r, c] = {
                    'blank': False,
                    'scores': {},
                    'raw': {},
                    'shape': None
                }

                continue

            masks = dict(variants)

            preferred = next(
                (
                    name
                    for name in (
                        'color_17',
                        'color_26',
                        'color_10',
                        'color_38',
                        'otsu'
                    )
                    if name in masks
                ),
                variants[0][0]
            )

            shape = normalize_mask(masks[preferred])
            votes = defaultdict(list)

            preferred_masks = [
                name
                for name in (
                    'color_17',
                    'color_26',
                    'color_10',
                    'otsu'
                )
                if name in masks
            ]

            if not preferred_masks:
                preferred_masks = [preferred]

            first = preferred_masks[0]

            # Initial OCR attempt
            answer = tesseract_one(masks[first], 10, 62)

            if answer:

                digit, confidence = answer
                votes[digit].append(confidence)

            strong = (
                answer is not None
                and answer[1] >= 88
            )

            # Retry alternative masks when needed
            if not strong:

                for name in preferred_masks[1:3]:

                    result_ocr = tesseract_one(
                        masks[name], 10, 70
                    )

                    if result_ocr:

                        digit, confidence = result_ocr
                        votes[digit].append(confidence)

                        if confidence >= 88:
                            strong = True
                            break

            # Alternative Tesseract segmentation mode
            if not strong:

                result_ocr = tesseract_one(
                    masks[first], 13, 70
                )

                if result_ocr:

                    digit, confidence = result_ocr
                    votes[digit].append(confidence)

            # Additional Otsu fallback
            if (
                not votes
                and 'otsu' in masks
                and first != 'otsu'
            ):

                result_ocr = tesseract_one(
                    masks['otsu'], 13, 76
                )

                if result_ocr:

                    digit, confidence = result_ocr
                    votes[digit].append(confidence)

            raw = {
                digit: max(confidences)
                for digit, confidences in votes.items()
            }

            scores = {
                digit: (
                    max(confidences)
                    + 2.0 * min(4, len(confidences))
                )
                for digit, confidences in votes.items()
            }

            details[r, c] = {
                'blank': False,
                'scores': scores,
                'raw': raw,
                'shape': shape,
                'variants': masks,
                'matches': {}
            }

    # Step 2: Build initial OCR guesses
    initial = [[0] * 9 for _ in range(9)]

    for (r, c), info in details.items():

        if info['scores']:
            initial[r][c] = max(
                info['scores'],
                key=info['scores'].get
            )

    doubtful = find_conflicts(initial)

    # Step 3: Collect reliable digit templates
    templates = []

    for (r, c), info in details.items():

        if (
            (r, c) in doubtful
            or info['shape'] is None
            or not info['raw']
        ):
            continue

        ranked = sorted(
            info['scores'].items(),
            key=lambda x: -x[1]
        )

        n, score = ranked[0]

        second = (
            ranked[1][1]
            if len(ranked) > 1
            else -999
        )

        if (
            info['raw'].get(n, 0) >= 82
            and score - second >= 10
        ):
            templates.append(
                (r, c, n, info['shape'])
            )

    # Step 4: Same-Sudoku digit comparison
    for (r, c), info in details.items():

        shape = info['shape']

        if shape is None:
            continue

        matches = {}

        for tr, tc, digit, refshape in templates:

            if (r, c) == (tr, tc):
                continue

            score = similarity(shape, refshape)

            matches[digit] = max(
                matches.get(digit, 0),
                score
            )

        info['matches'] = matches

        for digit, sim in matches.items():

            if sim >= 0.76:

                info['scores'][digit] = (
                    info['scores'].get(digit, 0)
                    + (sim - 0.62) * 170
                )

        # Very similar glyphs in the same Sudoku
        # may correct high-confidence OCR mistakes.
        shape_rank = sorted(
            matches.items(),
            key=lambda x: -x[1]
        )

        if shape_rank:

            label, lead = shape_rank[0]

            runnerup = (
                shape_rank[1][1]
                if len(shape_rank) > 1
                else 0.0
            )

            if lead >= 0.86 and lead - runnerup >= 0.10:

                highest = max(
                    info['scores'].values(),
                    default=0
                )

                info['scores'][label] = max(
                    info['scores'].get(label, 0),
                    highest + 12
                )

    # Step 5: Create provisional Sudoku board
    board = [[0] * 9 for _ in range(9)]

    for (r, c), info in details.items():

        if info['blank']:
            continue

        ranked = sorted(
            info['scores'].items(),
            key=lambda x: -x[1]
        )

        if not ranked:
            board[r][c] = None
            continue

        digit, score = ranked[0]

        if (
            info['raw'].get(digit, 0) >= MIN_OCR_CONF
            or info['matches'].get(digit, 0) >= 0.79
        ):
            board[r][c] = digit

        else:
            board[r][c] = None

    # Step 6: Resolve Sudoku conflicts conservatively
    for _ in range(30):

        conflicts = find_conflicts(board)

        if not conflicts:
            break

        weakest = min(
            conflicts,
            key=lambda rc: details[rc]['scores'].get(
                board[rc[0]][rc[1]],
                0
            )
        )

        r, c = weakest
        current = board[r][c]

        board[r][c] = None

        alternatives = sorted(
            details[r, c]['scores'].items(),
            key=lambda x: -x[1]
        )

        fixed = False

        for other, score in alternatives:

            if other == current or score < MIN_OCR_CONF:
                continue

            if legal_at(board, r, c, other):

                board[r][c] = other
                fixed = True
                break

        if not fixed:
            board[r][c] = None

    # Step 7: Identify unresolved cells
    review = sorted(
        (r, c)
        for r in range(9)
        for c in range(9)
        if board[r][c] is None
    )

    return board, review, details


def draw_board(board):
    """Display recognized Sudoku as a 9x9 image."""

    output = np.zeros(
        (SIDE, SIDE, 3),
        np.uint8
    )

    for r in range(9):
        for c in range(9):

            n = board[r][c]

            text = '?' if n is None else str(n)

            if n is None:
                color = (0, 0, 255)

            elif n == 0:
                color = (135, 135, 135)

            else:
                color = (0, 255, 0)

            cv2.putText(
                output,
                text,
                (c * CELL + 15, r * CELL + 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2
            )

    for i in range(10):

        width = 3 if i % 3 == 0 else 1

        cv2.line(
            output,
            (i * CELL, 0),
            (i * CELL, SIDE - 1),
            (0, 150, 0),
            width
        )

        cv2.line(
            output,
            (0, i * CELL),
            (SIDE - 1, i * CELL),
            (0, 150, 0),
            width
        )

    return output


def main(image_path, show_windows=True):

    # Configure Tesseract executable
    if Path(TESSERACT_PATH).is_file():

        pytesseract.pytesseract.tesseract_cmd = (
            TESSERACT_PATH
        )

    else:

        try:
            pytesseract.get_tesseract_version()

        except pytesseract.TesseractNotFoundError as error:

            raise RuntimeError(
                'Tesseract not found. Check TESSERACT_PATH.'
            ) from error

    # Step 1: Image processing
    result = process_sudoku(
        image_path,
        debug=True
    )

    # Step 2: Digit recognition
    board, review, details = recognize(result)

    # Step 3: Optional manual corrections
    for (r, c), digit in MANUAL_CORRECTIONS.items():

        if not (
            0 <= r < 9
            and 0 <= c < 9
            and type(digit) is int
            and 0 <= digit <= 9
        ):

            raise ValueError(
                f'Invalid manual correction: {(r, c)}->{digit}'
            )

        board[r][c] = digit

    # Step 4: Recalculate uncertain cells
    review = sorted(
        (r, c)
        for r in range(9)
        for c in range(9)
        if board[r][c] is None
    )

    # Step 5: Print Sudoku matrix
    print('\nRecognized Sudoku (0 empty, ? uncertain):')

    for row in board:

        print(
            ' '.join(
                '?' if n is None else str(n)
                for n in row
            )
        )

    print('\nReview needed:', review)

    # Print alternative candidates
    for r, c in review:

        candidates = sorted(
            details[r, c]['scores'].items(),
            key=lambda x: -x[1]
        )

        print(
            f'Cell ({r},{c}) candidates: '
            f'{candidates[:4]}'
        )

    # Step 6: Check remaining Sudoku conflicts
    remaining = find_conflicts(board)

    print('Remaining conflicts:', sorted(remaining))

    if review or remaining:

        print(
            'Do not solve yet: review unknown/conflicting '
            'cells against image.'
        )

    else:

        print(
            'No detected conflicts. Verify givens '
            'against image before solving.'
        )

    # Step 7: Display results
    if show_windows:

        cv2.imshow(
            'Straightened Sudoku',
            result.warped
        )

        cv2.imshow(
            'Cell Occupancy',
            __import__('image_processing').draw_status(result)
        )

        cv2.imshow(
            'Recognized Sudoku',
            draw_board(board)
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return board, review


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--image',
        type=str,
        default=str(DEFAULT_IMAGE)
    )

    parser.add_argument(
        '--no-windows',
        action='store_true'
    )

    args = parser.parse_args()

    main(
        args.image,
        show_windows=not args.no_windows
    )
