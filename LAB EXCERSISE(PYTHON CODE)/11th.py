
# Map Coloring Problem using CSP and Backtracking

# Define the map as a graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'C', 'D'],
    'C': ['A', 'B', 'D'],
    'D': ['B', 'C']
}

# Define the available colors
colors = ['Red', 'Green', 'Blue']

# Store the assigned colors
coloring = {}

# Check whether a color can be assigned to a region
def is_valid(region, color):
    for neighbor in graph[region]:
        if neighbor in coloring and coloring[neighbor] == color:
            return False
    return True

# Solve the CSP using backtracking
def map_coloring(regions):
    if not regions:
        return True

    region = regions[0]

    for color in colors:
        if is_valid(region, color):
            coloring[region] = color

            if map_coloring(regions[1:]):
                return True

            # Backtrack
            del coloring[region]

    return False

# Execute the program
regions = list(graph.keys())

if map_coloring(regions):
    print("Map Coloring Solution:")
    for region, color in coloring.items():
        print(region, ":", color)
else:
    print("No solution exists.")

