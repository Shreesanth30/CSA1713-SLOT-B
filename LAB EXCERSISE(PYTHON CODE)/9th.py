from itertools import permutations

def travelling_salesman(graph, start):
    n = len(graph)
    cities = list(range(n))
    cities.remove(start)

    min_cost = float('inf')
    best_path = []

    for path in permutations(cities):
        current_cost = 0
        k = start

        for city in path:
            current_cost += graph[k][city]
            k = city

        current_cost += graph[k][start]

        if current_cost < min_cost:
            min_cost = current_cost
            best_path = [start] + list(path) + [start]

    return min_cost, best_path


# Distance matrix representing cities
graph = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

cost, path = travelling_salesman(graph, 0)

print("Minimum cost:", cost)
print("Optimal path:", " -> ".join(map(str, path)))
