# Minimax Algorithm for Tic-Tac-Toe

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i], "|", board[i + 1], "|", board[i + 2])
        if i < 6:
            print("--+---+--")


def check_winner(board):
    win_patterns = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    for pattern in win_patterns:
        a, b, c = pattern
        if board[a] == board[b] == board[c]:
            return board[a]

    return None


def minimax(board, is_maximizing):
    winner = check_winner(board)

    if winner == "O":
        return 1
    elif winner == "X":
        return -1
    elif " " not in board:
        return 0

    if is_maximizing:
        best_score = -float("inf")

        for i in range(9):
            if board[i] == " ":
                board[i] = "O"
                score = minimax(board, False)
                board[i] = " "
                best_score = max(score, best_score)

        return best_score

    else:
        best_score = float("inf")

        for i in range(9):
            if board[i] == " ":
                board[i] = "X"
                score = minimax(board, True)
                board[i] = " "
                best_score = min(score, best_score)

        return best_score


def best_move(board):
    best_score = -float("inf")
    move = -1

    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            score = minimax(board, False)
            board[i] = " "

            if score > best_score:
                best_score = score
                move = i

    return move


# Main program
board = [" "] * 9

print("Tic-Tac-Toe: You are X, Computer is O")
print("Positions are numbered from 1 to 9.")

while True:
    print_board(board)

    try:
        position = int(input("Enter your move (1-9): ")) - 1
    except ValueError:
        print("Please enter a valid number.")
        continue

    if position < 0 or position > 8 or board[position] != " ":
        print("Invalid move! Try again.")
        continue

    board[position] = "X"

    if check_winner(board) == "X":
        print_board(board)
        print("You win!")
        brea

