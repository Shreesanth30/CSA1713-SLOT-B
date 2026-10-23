import heapq

graph = {
    "Gate": {"Library": 120, "Canteen": 200},
    "Library": {"Gate": 120, "Lab": 150, "Hostel": 300},
    "Canteen": {"Gate": 200, "Hostel": 100, "Admin": 260},
    "Lab": {"Library": 150, "Auditorium": 350},
    "Hostel": {"Library": 300, "Canteen": 100, "Auditorium": 180},
    "Admin": {"Canteen": 260, "Auditorium": 120},
    "Auditorium": {"Lab": 350, "Hostel": 180, "Admin": 120}
}

h = {
    "Gate": 400,
    "Library": 330,
    "Canteen": 240,
    "Lab": 280,
    "Hostel": 160,
    "Admin": 110,
    "Auditorium": 0
}


def greedy_best_first(start, goal):
    frontier = [(h[start], start, [start])]
    visited = set()

    while frontier:
        _, node, path = heapq.heappop(frontier)

        if node == goal:
            return path

        if node in visited:
            continue

        visited.add(node)

        for neighbour in graph[node]:
            if neighbour not in visited:
                heapq.heappush(
                    frontier,
                    (h[neighbour], neighbour, path + [neighbour])
                )

    return None


def a_star(start, goal):
    frontier = [(h[start], 0, start, [start])]
    best_cost = {start: 0}

    while frontier:
        f, g, node, path = heapq.heappop(frontier)

        if g != best_cost.get(node):
            continue

        if node == goal:
            return path, g

        for neighbour, distance in graph[node].items():
            new_g = g + distance

            if new_g < best_cost.get(neighbour, float("inf")):
                best_cost[neighbour] = new_g
                new_f = new_g + h[neighbour]

                heapq.heappush(
                    frontier,
                    (new_f, new_g, neighbour, path + [neighbour])
                )

    return None, float("inf")


greedy_path = greedy_best_first("Gate", "Auditorium")
print("Greedy path:", " -> ".join(greedy_path))

astar_path, astar_distance = a_star("Gate", "Auditorium")
print("A* path:", " -> ".join(astar_path))
print("A* distance:", astar_distance, "metres")
