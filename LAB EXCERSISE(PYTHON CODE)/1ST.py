import heapq

# Goal state
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)

# Possible movements of blank tile
MOVES = {
    0: [1, 3],
    1: [0, 2, 4],
    2: [1, 5],
    3: [0, 4, 6],
    4: [1, 3, 5, 7],
    5: [2, 4, 8],
    6: [3, 7],
    7: [4, 6, 8],
    8: [5, 7]
}

# Manhattan Distance heuristic
def heuristic(state):
    distance = 0

    for i, value in enumerate(state):
        if value != 0:
            goal_index = GOAL.index(value)

            x1, y1 = divmod(i, 3)
            x2, y2 = divmod(goal_index, 3)

            distance += abs(x1 - x2) + abs(y1 - y2)

    return distance


# Generate possible next states
def get_neighbors(state):
    blank = state.index(0)

    for move in MOVES[blank]:
        new_state = list(state)

        # Swap blank with adjacent tile
        new_state[blank], new_state[move] = \
            new_state[move], new_state[blank]

        yield tuple(new_state)


# A* Search Algorithm
def solve(start):
    priority_queue = []

    # (f, g, state, path)
    heapq.heappush(
        priority_queue,
        (heuristic(start), 0, start, [start])
    )

    visited = set()

    while priority_queue:

        f, g, state, path = heapq.heappop(priority_queue)

        if state in visited:
            continue

        visited.add(state)

        # Goal reached
        if state == GOAL:
            return path

        for neighbor in get_neighbors(state):

            if neighbor not in visited:
                new_g = g + 1
                new_f = new_g + heuristic(neighbor)

                heapq.heappush(
                    priority_queue,
                    (new_f, new_g, neighbor, path + [neighbor])
                )

    return None


# Display puzzle
def display(state):
    for i in range(0, 9, 3):
        print(state[i:i+3])
    print()


# Main program
start = (
    1, 2, 3,
    4, 0, 6,
    7, 5, 8
)

print("Initial State:")
display(start)

solution = solve(start)

if solution:
    print("Solution found!")
    print("Number of moves:", len(solution) - 1)

    for step, state in enumerate(solution):
        print("Step", step)
        display(state)
else:
    print("No solution exists.")
