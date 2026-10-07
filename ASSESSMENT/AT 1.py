from collections import deque

SIZE = 5
START = (1, 1)
GOAL = (5, 5)

BLOCKED = {
    (2, 2), (2, 3), (3, 3),
    (4, 2), (4, 4)
}

# Neighbor order used in the written BFS and DFS traces.
MOVES = [
    ("Up",    (-1, 0)),
    ("Down",  (1, 0)),
    ("Left",  (0, -1)),
    ("Right", (0, 1))
]

def neighbors(cell):
    row, col = cell

    for action, (dr, dc) in MOVES:
        next_cell = (row + dr, col + dc)
        next_row, next_col = next_cell

        inside_grid = (
            1 <= next_row <= SIZE and
            1 <= next_col <= SIZE
        )

        if inside_grid and next_cell not in BLOCKED:
            yield action, next_cell

def reconstruct_path(parent, state):
    path = []

    while state in parent:
        previous_state, action = parent[state]
        path.append((action, state))
        state = previous_state

    return list(reversed(path))

def bfs():
    frontier = deque([START])
    discovered = {START}
    parent = {}

    while frontier:
        current = frontier.popleft()

        if current == GOAL:
            return reconstruct_path(parent, GOAL)

        for action, next_cell in neighbors(current):
            if next_cell not in discovered:
                discovered.add(next_cell)
                parent[next_cell] = (current, action)
                frontier.append(next_cell)

    return None

def dfs():
    stack = [START]
    discovered = {START}
    parent = {}

    while stack:
        current = stack.pop()

        if current == GOAL:
            return reconstruct_path(parent, GOAL)

        # Reverse the order so Up is explored first.
        for action, next_cell in reversed(list(neighbors(current))):
            if next_cell not in discovered:
                discovered.add(next_cell)
                parent[next_cell] = (current, action)
                stack.append(next_cell)

    return None

for name, search in [("BFS", bfs), ("DFS", dfs)]:
    path = search()

    print(f"\n{name}")

    if path is None:
        print("No path found.")
    else:
        print("Start:", START)
        for action, cell in path:
            print(f"{action} -> {cell}")
        print("Total cost:", len(path))
