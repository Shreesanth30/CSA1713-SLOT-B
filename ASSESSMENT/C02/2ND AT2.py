def minimax(node, maximizing):
    if isinstance(node, int):
        return node

    values = [minimax(child, not maximizing) for child in node]

    if maximizing:
        return max(values)

    return min(values)


game_tree = [
    [[4, 7], [8, 2]],
    [[3, 5], [6, 9]]
]

result = minimax(game_tree, True)

print("Best guaranteed score for X:", result)
