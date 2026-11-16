
from digit_recognition import main as recognize_sudoku
from digit_recognition import find_conflicts
from image_processing import DEFAULT_IMAGE

# Step 1: Get the board from digit recognition
board, review = recognize_sudoku(
    DEFAULT_IMAGE,
    show_windows=False
)

# Step 2: Check whether recognition succeeded
if review or find_conflicts(board):
    print("Cannot identify Sudoku")
    board = None
else:
    print("Sudoku recognized successfully!")


def empty(board):
    candidates = {}
    boxes = {}
    for row in range(9):  # Step 1: Find candidates for all empty cells
        for col in range(9):
            if board[row][col] == 0:
                possible = []
                for num in range(1, 10):
                    if is_valid(board, row, col, num):
                        possible.append(num)
                candidates[(row, col)] = possible
                if len(possible) == 0:  # If no number is possible, backtrack
                    return (row, col, [])
                b_row = (row // 3) * 3  # Step 2: Group cells into 3x3 boxes
                b_col = (col // 3) * 3
                box = (b_row, b_col)
                if box not in boxes:
                    boxes[box] = []
                boxes[box].append((row, col))
    # Step 3: Check Hidden Singles first
    for cells in boxes.values():
        for r, c in cells:
            for num in candidates[(r, c)]:
                count = 0
                for other_r, other_c in cells:
                    if num in candidates[(other_r, other_c)]:
                        count += 1
                if count == 1:
                    return (r, c, [num])
    # Step 4: If no Hidden Single, use MRV
    best_cell = None
    min_count = 10
    for (row, col), possible in candidates.items():
        if len(possible) < min_count:
            min_count = len(possible)
            best_cell = (row, col, possible)
    return best_cell

def is_valid(board, row, col,num):
    for i in range(9):
        if board[row][i] == num:
            return False
    for i in range(9):
        if board[i][col] == num:
            return False
    b_row = (row // 3)*3
    b_col = (col // 3)*3
    for i in range(b_row,b_row+3):
        for j in range(b_col,b_col+3):
            if board[i][j] == num:
                return False
    return True
def solve(board):
    emptycell = empty(board)
    if emptycell is None:
        return True
    r,c,possible = emptycell
    for num in possible:
            board[r][c] = num
            if solve(board):
                return True
            board[r][c] = 0
    return False

if __name__ == "__main__":
    if board is not None:
        if solve(board):
            print("Sudoku solved successfully!")
        else:
            print("No Sudoku solution found!")
