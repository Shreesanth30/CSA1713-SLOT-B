N = 4

board = [[0 for i in range(N)] for j in range(N)]


def is_safe(row, col):
    for i in range(row):
        if board[i][col] == 1:
            return False

    i, j = row - 1, col - 1
    while i >= 0 and j >= 0:
        if board[i][j] == 1:
            return False
        i -= 1
        j -= 1

    i, j = row - 1, col + 1
    while i >= 0 and j < N:
        if board[i][j] == 1:
            return False
        i -= 1
        j += 1

    return True


def solve(row):
    if row == N:
        print_board()
        return

    for col in range(N):
        if is_safe(row, col):
            board[row][col] = 1
            solve(row + 1)
            board[row][col] = 0


def print_board():
    print("Solution:")
    for row in board:
        print(row)
    print()


solve(0)
