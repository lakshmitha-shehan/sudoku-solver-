from sudoku_solver import board, solve
import tkinter as tk
if solve(board):
    print("Sudoku solved successfully!")
else:
    raise ValueError("No Sudoku solution found!")

root = tk.Tk()
root.title("Sudoku Solver")
root.geometry("510x510")
root.configure(bg="#000000")
# Canvas
canvas = tk.Canvas(
    root,
    width=470,
    height=470,
    bg="#000000",
    highlightthickness=0
)
canvas.pack(pady=20)
cell_size = 50
padding = 10
for i in range(10):
    line_width = 3 if i % 3 == 0 else 1
    line_color = "white" if i % 3 == 0 else "green"
    position = padding + i * cell_size
    # Vertical lines
    canvas.create_line(
        position, padding,
        position, padding + 450,
        width=line_width,
        fill=line_color
    )
    # Horizontal lines
    canvas.create_line(
        padding, position,
        padding + 450, position,
        width=line_width,
        fill=line_color
    )

for row in range(9):
    for col in range(9):
        x = padding + col * cell_size + cell_size / 2
        y = padding + row * cell_size + cell_size / 2
        canvas.create_text(
            x, y,
            text=str(board[row][col]),
            font=("Consolas", 22, "bold"),
            fill="green"
        )

root.mainloop()
