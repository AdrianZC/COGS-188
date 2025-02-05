def print_board(board):
    """
    Prints the Sudoku board in a grid format.
    0 indicates an empty cell.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    None
    """
    for row_idx, row in enumerate(board):
        # Print a horizontal separator every 3 rows (for sub-grids)
        if row_idx % 3 == 0 and row_idx != 0:
            print("- - - - - - - - - - -")

        row_str = ""
        for col_idx, value in enumerate(row):
            # Print a vertical separator every 3 columns (for sub-grids)
            if col_idx % 3 == 0 and col_idx != 0:
                row_str += "| "

            if value == 0:
                row_str += ". "
            else:
                row_str += str(value) + " "
        print(row_str.strip())


def find_empty_cell(board):
    """
    Finds an empty cell (indicated by 0) in the Sudoku board.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 represents an empty cell.

    Returns:
    tuple or None:
        - If there is an empty cell, returns (row_index, col_index).
        - If there are no empty cells, returns None.
    """
    for row in range(9):
        for col in range(9):
            if board[row][col] == 0:
                return row, col
            
    return None


def is_valid(board, row, col, num):
    """
    Checks if placing 'num' at board[row][col] is valid under Sudoku rules:
    1) 'num' is not already in the same row
    2) 'num' is not already in the same column
    3) 'num' is not already in the 3x3 sub-box containing that cell

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.
    row (int): Row index of the cell.
    col (int): Column index of the cell.
    num (int): The candidate number to place.

    Returns:
    bool: True if valid, False otherwise.
    """
    if not isinstance(num, int) or not 1 <= num <= 9:
        raise ValueError("Number must be between 1 and 9")
    
    if not isinstance(row, int) or not isinstance(col, int):
        raise ValueError("Row and column must be integers")
        
    if not 0 <= row < 9 or not 0 <= col < 9:
        raise ValueError("Row and column must be between 0 and 8")
        
    if not isinstance(board, list) or len(board) != 9 or not all(len(row) == 9 for row in board):
        raise ValueError("Invalid board format - must be 9x9")
    
    for x in range(9):
        if board[row][x] == num and x != col:
            return False
        
    for x in range(9):
        if board[x][col] == num and x != row:
            return False
        
    box_row = (row // 3) * 3
    box_col = (col // 3) * 3

    for x in range(box_row, box_row + 3):
        for y in range(box_col, box_col + 3):
            if board[x][y] == num and (x, y) != (row, col):
                return False

    return True


def solve_sudoku(board):
    """
    Solves the Sudoku puzzle in 'board' using backtracking.

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board where 0 indicates an empty cell.

    Returns:
    bool:
        - True if the puzzle is solved successfully.
        - False if the puzzle is unsolvable.
    """
    empty = find_empty_cell(board)
    
    if not empty:
        return True
    
    row, col = empty
    
    for num in range(1, 10):
        if is_valid(board, row, col, num):
            board[row][col] = num
            
            if solve_sudoku(board):
                return True
            
            board[row][col] = 0
    
    return False


def is_solved_correctly(board):
    """
    Checks that the board is fully and correctly solved:
    - Each row contains digits 1-9 exactly once
    - Each column contains digits 1-9 exactly once
    - Each 3x3 sub-box contains digits 1-9 exactly once

    Parameters:
    board (list[list[int]]): A 9x9 Sudoku board.

    Returns:
    bool: True if the board is correctly solved, False otherwise.
    """
    if any(0 in row for row in board):
        return False
        
    for row in board:
        if len(set(row)) != 9:
            return False
    
    for col in range(9):
        column = [board[row][col] for row in range(9)]
        if len(set(column)) != 9:
            return False
    
    for box_row in range(0, 9, 3):
        for box_col in range(0, 9, 3):
            box = []
            for i in range(3):
                for j in range(3):
                    box.append(board[box_row + i][box_col + j])
            if len(set(box)) != 9:
                return False
                
    return True


if __name__ == "__main__":
    # Example usage / debugging:
    example_board = [
        [7, 8, 0, 4, 0, 0, 1, 2, 0],
        [6, 0, 0, 0, 7, 5, 0, 0, 9],
        [0, 0, 0, 6, 0, 1, 0, 7, 8],
        [0, 0, 7, 0, 4, 0, 2, 6, 0],
        [0, 0, 1, 0, 5, 0, 9, 3, 0],
        [9, 0, 4, 0, 6, 0, 0, 0, 5],
        [0, 7, 0, 3, 0, 0, 0, 1, 2],
        [1, 2, 0, 0, 0, 7, 4, 0, 0],
        [0, 4, 9, 2, 0, 6, 0, 0, 7],
    ]

    print("Debug: Original board:\n")
    print_board(example_board)
    # TODO: Students can call their solve_sudoku here once implemented and check if they got a correct solution.
    solve_sudoku(example_board)
    print("\nDebug: Solved board:\n")
    print_board(example_board)