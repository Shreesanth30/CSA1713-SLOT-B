
import heapq

def astar(graph, heuristic, start, goal):
    # Priority queue: (f_score, current_node)
    open_list = [(0, start)]

    # Store the cost to reach each node
    g_score = {start: 0}

    # Store the parent of each node
    came_from = {}

    while open_list:
        _, current = heapq.heappop(open_list)

        # Check if the goal is reached
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal]

        # Explore neighboring nodes
        for neighbor, cost in graph[current]:
            tentative_g = g_score[current] + cost

            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g

                # f(n) = g(n) + h(n)
                f_score = tentative_g + heuristic[neighbor]
                heapq.heappush(open_list, (f_score, neighbor))

    return None, float("inf")


# Graph represented as an adjacency list
graph = {
    'A': [('B', 1), ('C', 4)],
    'B': [('A', 1), ('C', 2), ('D', 5)],
    'C': [('A', 4), ('B', 2), ('D', 1)],
    'D': [('B', 5), ('C', 1)]
}

# Heuristic values for each node
heuristic = {
    'A': 4,
    'B': 3,
    'C': 1,
    'D': 0
}

start = 'A'
goal = 'D'

path, cost = astar(graph, heuristic, start, goal)

if path:
    print("Shortest path:", " -> ".join(path))
    print("Total cost:", cost)
else:
    print("No path found.")

