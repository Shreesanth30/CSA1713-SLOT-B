
def print_board(board):
    print()
    for i in range(3):
        print(" " + " | ".join(board[i * 3:(i + 1) * 3]))
        if i < 2:
            print("---+---+---")
    print()


def check_winner(board):
    winning_combinations = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),
        (0, 3, 6), (1, 4, 7), (2, 5, 8),
        (0, 4, 8), (2, 4, 6)
    ]

    for a, b, c in winning_combinations:
        if board[a] == board[b] == board[c]:
            return True

    return False


def tic_tac_toe():
    board = [str(i) for i in range(1, 10)]
    current_player = "X"

    print("Welcome to Tic-Tac-Toe!")
    print("Enter a number from 1 to 9 to place your symbol.")

    while True:
        print_board(board)

        try:
            move = int(input(f"Player {current_player}, enter your move: "))
        except ValueError:
            print("Invalid input! Please enter a number.")
            continue

        if move < 1 or move > 9:
            print("Please enter a number between 1 and 9.")
            continue

        if board[move - 1] in ("X", "O"):
            print("That position is already occupied! Try again.")
            continue

        board[move - 1] = current_player

        if check_winner(board):
            print_board(board)
            print(f"Congratulations! Player {current_player} wins!")
            break

        if all(cell in ("X", "O") for cell in board):
            print_board(board)
            print("It's a draw!")
            break

        current_player = "O" if current_player == "X" else "X"


tic_tac_toe()

