def dls(graph, node, goal, depth, path):
    if node == goal:
        print("Path:", " -> ".join(path))
        return True

    if depth <= 0:
        return False

    for child in graph[node]:
        if child not in path:
            if dls(graph, child, goal, depth - 1, path + [child]):
                return True

    return False


def iddfs(graph, start, goal):
    for depth in range(len(graph)):
        print("Depth Limit:", depth)
        if dls(graph, start, goal, depth, [start]):
            return
    print("Goal not found")


graph = {
    'A': ['B', 'C'],
    'B': ['D', 'E'],
    'C': ['F', 'G'],
    'D': [],
    'E': [],
    'F': [],
    'G': []
}

iddfs(graph, 'A', 'G')
