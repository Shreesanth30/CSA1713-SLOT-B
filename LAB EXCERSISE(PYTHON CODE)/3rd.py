from collections import deque

def water_jug_problem(capacity1, capacity2, target):
    visited = set()
    queue = deque()

    # Initial state
    queue.append((0, 0, []))

    while queue:
        jug1, jug2, path = queue.popleft()

        if (jug1, jug2) in visited:
            continue

        visited.add((jug1, jug2))

        # Add current state to path
        path = path + [(jug1, jug2)]

        # Check whether target is reached
        if jug1 == target or jug2 == target:
            print("Solution found:")
            for state in path:
                print(state)
            return

        # Generate possible states
        states = [
            (capacity1, jug2),  # Fill Jug 1
            (jug1, capacity2),  # Fill Jug 2
            (0, jug2),          # Empty Jug 1
            (jug1, 0),          # Empty Jug 2
            # Pour Jug 1 into Jug 2
            (max(0, jug1 - (capacity2 - jug2)),
             min(capacity2, jug1 + jug2)),
            # Pour Jug 2 into Jug 1
            (min(capacity1, jug1 + jug2),
             max(0, jug2 - (capacity1 - jug1)))
        ]

        for state in states:
            if state not in visited:
                queue.append((state[0], state[1], path))


# Example:
# Jug 1 = 4 litres
# Jug 2 = 3 litres
# Target = 2 litres

water_jug_problem(4, 3, 2)
