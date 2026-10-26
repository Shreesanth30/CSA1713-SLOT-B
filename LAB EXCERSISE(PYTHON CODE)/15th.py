
def alpha_beta(depth, node, maximizing_player, values,
               alpha, beta, max_depth):
    # Base case: Leaf node
    if depth == max_depth:
        return values[node]

    if maximizing_player:
        best = float('-inf')

        # Explore child nodes
        for i in range(2):
            val = alpha_beta(
                depth + 1,
                node * 2 + i,
                False,
                values,
                alpha,
                beta,
                max_depth
            )

            best = max(best, val)
            alpha = max(alpha, best)

            # Beta pruning
            if beta <= alpha:
                break

        return best

    else:
        best = float('inf')

        # Explore child nodes
        for i in range(2):
            val = alpha_beta(
                depth + 1,
                node * 2 + i,
                True,
                values,
                alpha,
                beta,
                max_depth
            )

            best = min(best, val)
            beta = min(beta, best)

            # Alpha pruning
            if beta <= alpha:
                break

        return best


# Main program
values = [3, 5, 6, 9, 1, 2, 0, -1]
max_depth = 3

result = alpha_beta(
    0, 0, True, values,
    float('-inf'),
    float('inf'),
    max_depth
)

print("The optimal value is:", result)

