from collections import deque

# State: (missionaries on left bank, cannibals on left bank, boat side)
# boat_side: 0 = left bank, 1 = right bank

TOTAL_MISSIONARIES = 3
TOTAL_CANNIBALS = 3
BOAT_CAPACITY = 2


def is_safe(missionaries_left, cannibals_left):
    missionaries_right = TOTAL_MISSIONARIES - missionaries_left
    cannibals_right = TOTAL_CANNIBALS - cannibals_left

    # Cannibals must not outnumber missionaries on either bank
    if missionaries_left > 0 and cannibals_left > missionaries_left:
        return False
    if missionaries_right > 0 and cannibals_right > missionaries_right:
        return False

    return True


def get_moves():
    # Possible numbers of missionaries and cannibals in the boat
    return [
        (missionaries, cannibals)
        for missionaries in range(BOAT_CAPACITY + 1)
        for cannibals in range(BOAT_CAPACITY + 1)
        if 1 <= missionaries + cannibals <= BOAT_CAPACITY
    ]


def solve():
    start = (TOTAL_MISSIONARIES, TOTAL_CANNIBALS, 0)
    goal = (0, 0, 1)

    queue = deque([start])
    previous = {start: None}

    while queue:
        state = queue.popleft()

        if state == goal:
            break

        missionaries_left, cannibals_left, boat_side = state

        for missionaries, cannibals in get_moves():
            if boat_side == 0:
                # Boat travels from left to right
                next_state = (
                    missionaries_left - missionaries,
                    cannibals_left - cannibals,
                    1,
                )
            else:
                # Boat travels from right to left
                next_state = (
                    missionaries_left + missionaries,
                    cannibals_left + cannibals,
                    0,
                )

            new_missionaries_left, new_cannibals_left, _ = next_state

            if not (0 <= new_missionaries_left <= TOTAL_MISSIONARIES):
                continue
            if not (0 <= new_cannibals_left <= TOTAL_CANNIBALS):
                continue
            if not is_safe(new_missionaries_left, new_cannibals_left):
                continue
            if next_state in previous:
                continue

            previous[next_state] = state
            queue.append(next_state)

    if goal not in previous:
        print("No solution found.")
        return

    # Reconstruct the path from the goal back to the start
    path = []
    state = goal
    while state is not None:
        path.append(state)
        state = previous[state]
    path.reverse()

    # Display each crossing
    for step, (missionaries_left, cannibals_left, boat_side) in enumerate(path):
        missionaries_right = TOTAL_MISSIONARIES - missionaries_left
        cannibals_right = TOTAL_CANNIBALS - cannibals_left
        boat = "Left" if boat_side == 0 else "Right"

        print(
            f"Step {step}: "
            f"Left bank: {missionaries_left}M {cannibals_left}C | "
            f"Right bank: {missionaries_right}M {cannibals_right}C | "
            f"Boat: {boat}"
        )


if __name__ == "__main__":
    solve()
