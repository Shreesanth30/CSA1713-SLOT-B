def alpha_beta(node, alpha, beta, maximizing):
    if isinstance(node, int):
        return node

    if maximizing:
        value = float("-inf")

        for child in node:
            value = max(
                value,
                alpha_beta(child, alpha, beta, False)
            )
            alpha = max(alpha, value)

            if alpha >= beta:
                break

        return value

    else:
        value = float("inf")

        for child in node:
            value = min(
                value,
                alpha_beta(child, alpha, beta, True)
            )
            beta = min(beta, value)

            if beta <= alpha:
                break

        return value


moves = [
    [6, 9, 7],
    [4, 8, 10],
    [7, 5, 12]
]

best_score = alpha_beta(
    moves,
    float("-inf"),
    float("inf"),
    True
)

print("Best move value:", best_score)
