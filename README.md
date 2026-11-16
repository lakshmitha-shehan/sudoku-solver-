# 🧩 Sudoku Solver – Python, OpenCV & Tesseract OCR

A Python-based Sudoku Solver that combines **Computer Vision, Optical Character Recognition (OCR), and algorithmic problem-solving** to recognize Sudoku puzzles from images, solve them, and display the completed puzzle using a Tkinter graphical interface.

The project takes a Sudoku image as input, detects the grid, recognizes the given digits, converts them into a 9×9 numerical matrix, and solves the puzzle using a combination of constraint-based techniques and recursive backtracking.

## ✨ Features

### 1. Image Processing & Grid Detection

- Detects Sudoku grids from input images.
- Performs image preprocessing and perspective correction.
- Extracts individual cells from the Sudoku grid.
- Identifies empty and occupied cells.

### 2. Digit Recognition

- Uses Tesseract OCR to recognize Sudoku digits.
- Processes individual cell images for digit recognition.
- Converts recognized digits into a 9×9 numerical matrix.
- Represents empty cells using `0`.

### 3. Sudoku Solving Algorithm

The solver combines three techniques:

- **Hidden Singles:** Identifies numbers that have only one possible position within a row, column, or 3×3 box.
- **MRV (Minimum Remaining Values):** Selects the empty cell with the fewest valid candidates to reduce the search space.
- **Recursive Backtracking:** Tries possible candidate values and backtracks when a contradiction occurs.

### 4. Graphical User Interface

- Developed using Tkinter.
- Displays the final solved Sudoku grid.
- Custom dark coding-style interface.
- Uses different line thicknesses to distinguish 3×3 subgrids.

## 🛠️ Technologies Used

| Technology    | Purpose                              |
| ------------- | ------------------------------------ |
| Python        | Main programming language            |
| OpenCV        | Image processing and grid detection  |
| Tesseract OCR | Digit recognition                    |
| NumPy         | Numerical and image array operations |
| Tkinter       | Graphical user interface             |

## ⚙️ How It Works

```text
Sudoku Image (PNG / JPG)
          |
          v
    Image Processing
         (OpenCV)
          |
          v
     Grid Detection
          |
          v
      Cell Extraction
          |
          v
     Digit Recognition
       (Tesseract)
          |
          v
     9 × 9 Sudoku Matrix
          |
          v
    Sudoku Solver
          |
     +----+----+
     |         |
 Hidden      MRV
 Singles      |
     |         |
     +----+----+
          |
    Backtracking
          |
          v
    Solved Sudoku
          |
          v
      Tkinter GUI
```

## 🚀 Installation & Setup

### Prerequisites

- Python 3.x
- Tesseract OCR
- pip

## ⚠️ Current Limitations

- Recognition accuracy depends on image quality, lighting, and contrast.
- Handwritten digits and unusual fonts may cause OCR errors.
- Skewed, distorted, or noisy images can affect grid detection.
- Incorrect OCR results can lead to an invalid Sudoku matrix or an unsolvable puzzle.
- Currently designed for standard 9×9 Sudoku puzzles.

## 🤖 AI Assistance

ChatGPT was used as a learning and development assistant, particularly for understanding and implementing parts of the Tesseract OCR digit recognition module.

The project also provided an opportunity to explore computer vision, OCR integration, algorithm design, debugging, and testing.

## 📚 Learning Outcomes

Through this project, I explored:

- Computer Vision and image preprocessing.
- Optical Character Recognition.
- Grid detection and cell extraction.
- Constraint-based problem-solving.
- Minimum Remaining Values (MRV).
- Hidden Singles and Recursive Backtracking.
- Python GUI development and software integration.

---

**Developed as a Python learning project.**

