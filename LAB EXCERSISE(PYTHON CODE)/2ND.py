# 8-Queens Problem using Backtracking

N = 8

# Check whether a queen can be placed at board[row][col]
def is_safe(board, row, col):
    # Check the same column
    for i in range(row):
        if board[i] == col:
            return False

    # Check upper-left diagonal
    for i, j in zip(range(row - 1, -1, -1), range(col - 1, -1, -1)):
        if board[i] == j:
            return False

    # Check upper-right diagonal
    for i, j in zip(range(row - 1, -1, -1), range(col + 1, N)):
        if board[i] == j:
            return False

    return True


# Solve the problem using backtracking
def solve(board, row):
    if row == N:
        print_board(board)
        return True

    for col in range(N):
        if is_safe(board, row, col):
            board[row] = col

            if solve(board, row + 1):
                return True

            # Backtrack
            board[row] = -1

    return False


# Display the chess board
def print_board(board):
    for row in range(N):
        for col in range(N):
            if board[row] == col:
                print("Q", end=" ")
            else:
                print(".", end=" ")
        print()


# Main program
board = [-1] * N

if not solve(board, 0):
    print("No solution exists.")
