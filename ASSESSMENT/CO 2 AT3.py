"""
CSA17 - Artificial Intelligence | Assessment Tool 3 (CO2)
Python code that reproduces the dry-run answers for Questions 1-5.

Run:  python ai_assessment_solutions.py
"""

# ---------------------------------------------------------------------------
# Data for Questions 1 and 2
# ---------------------------------------------------------------------------
EDGES = [("S", "A", 2), ("S", "B", 3), ("A", "C", 4), ("A", "D", 7),
         ("B", "D", 2), ("B", "E", 6), ("C", "G", 6), ("D", "G", 3),
         ("E", "G", 2)]
H = {"S": 7, "A": 8, "B": 4, "C": 5, "D": 3, "E": 2, "G": 0}

GRAPH = {n: [] for n in H}
for u, v, c in EDGES:
    GRAPH[u].append((v, c))
    GRAPH[v].append((u, c))


def build_path(parent, goal):
    path, node = [], goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]


# ---------------------------------------------------------------------------
# Question 1: Greedy Best-First Search  (priority = h(n))
# ---------------------------------------------------------------------------
def greedy_best_first(start="S", goal="G"):
    print("=" * 60)
    print("QUESTION 1: GREEDY BEST-FIRST SEARCH")
    print("=" * 60)
    open_list = {start: H[start]}
    parent = {start: None}
    g = {start: 0}
    closed = set()
    step = 0
    while open_list:
        node = min(open_list, key=lambda n: (open_list[n], n))
        open_list.pop(node)
        closed.add(node)
        step += 1
        if node == goal:
            shown = ", ".join(f"{n}:{h}" for n, h in sorted(open_list.items(), key=lambda x: x[1]))
            print(f"Step {step}: expand {node}  -> GOAL reached | open = {shown}")
            break
        for nb, cost in GRAPH[node]:
            if nb not in closed and nb not in open_list:
                open_list[nb] = H[nb]
                parent[nb] = node
                g[nb] = g[node] + cost
        shown = ", ".join(f"{n}:{h}" for n, h in sorted(open_list.items(), key=lambda x: x[1]))
        print(f"Step {step}: expand {node}  -> open = {shown}")
    path = build_path(parent, goal)
    print("Path:", " -> ".join(path), "| cost =", g[goal])
    return path, g[goal]


# ---------------------------------------------------------------------------
# Question 2: A* Search  (priority = f(n) = g(n) + h(n))
# ---------------------------------------------------------------------------
def a_star(start="S", goal="G"):
    print("=" * 60)
    print("QUESTION 2: A* SEARCH")
    print("=" * 60)
    g = {start: 0}
    parent = {start: None}
    open_list = {start}
    closed = set()
    step = 0
    while open_list:
        node = min(open_list, key=lambda n: (g[n] + H[n], H[n], n))
        open_list.remove(node)
        closed.add(node)
        step += 1
        if node != goal:
            for nb, cost in GRAPH[node]:
                new_g = g[node] + cost
                if nb in closed and new_g >= g.get(nb, float("inf")):
                    continue
                if new_g < g.get(nb, float("inf")):
                    g[nb] = new_g
                    parent[nb] = node
                    open_list.add(nb)
        shown = ", ".join(f"{n}:{g[n]}+{H[n]}={g[n] + H[n]}"
                          for n in sorted(open_list, key=lambda n: (g[n] + H[n], n)))
        tag = "  -> GOAL reached" if node == goal else ""
        print(f"Step {step}: expand {node} (f={g[node] + H[node]}){tag} | open = {shown}")
        if node == goal:
            break
    path = build_path(parent, goal)
    print("Path:", " -> ".join(path), "| cost =", g[goal])
    return path, g[goal]


def check_admissibility():
    """True remaining cost h*(n) by Dijkstra from the goal, compared with h(n)."""
    import heapq
    dist = {n: float("inf") for n in H}
    dist["G"] = 0
    pq = [(0, "G")]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, c in GRAPH[u]:
            if d + c < dist[v]:
                dist[v] = d + c
                heapq.heappush(pq, (dist[v], v))
    print("\nAdmissibility check (h(n) <= h*(n)):")
    for n in ["S", "A", "B", "C", "D", "E", "G"]:
        print(f"  {n}: h = {H[n]}, true cost h* = {dist[n]}, admissible = {H[n] <= dist[n]}")


# ---------------------------------------------------------------------------
# Question 3: CSP - backtracking + forward checking + MRV + degree
# ---------------------------------------------------------------------------
VARS = ["WA", "NT", "SA", "Q", "NSW", "V"]
COLOURS = ["R", "G", "B"]
CONSTRAINTS = [("WA", "NT"), ("WA", "SA"), ("NT", "SA"), ("NT", "Q"), ("SA", "Q"),
               ("SA", "NSW"), ("SA", "V"), ("Q", "NSW"), ("NSW", "V")]
NEIGH = {v: [] for v in VARS}
for a, b in CONSTRAINTS:
    NEIGH[a].append(b)
    NEIGH[b].append(a)


def select_variable(domains, assignment):
    """MRV, ties broken by degree (most unassigned neighbours), then fixed order."""
    unassigned = [v for v in VARS if v not in assignment]

    def key(v):
        degree = sum(1 for n in NEIGH[v] if n not in assignment)
        return (len(domains[v]), -degree, VARS.index(v))
    return min(unassigned, key=key)


def backtrack(assignment, domains, counter):
    if len(assignment) == len(VARS):
        return assignment
    var = select_variable(domains, assignment)
    for val in domains[var]:
        counter[0] += 1
        new_dom = {v: list(d) for v, d in domains.items()}
        new_dom[var] = [val]
        wipe_out = False
        for n in NEIGH[var]:
            if n not in assignment and val in new_dom[n]:
                new_dom[n].remove(val)
                if not new_dom[n]:
                    wipe_out = True
        rest = {v: "".join(new_dom[v]) for v in VARS if v not in assignment and v != var}
        print(f"Assignment {counter[0]}: {var} = {val} | updated domains: {rest}"
              + ("  -> DEAD END, backtrack" if wipe_out else ""))
        if not wipe_out:
            result = backtrack({**assignment, var: val}, new_dom, counter)
            if result:
                return result
    return None


def solve_csp():
    print("=" * 60)
    print("QUESTION 3: CSP (backtracking + forward checking + MRV + degree)")
    print("=" * 60)
    domains = {v: list(COLOURS) for v in VARS}
    solution = backtrack({}, domains, [0])
    print("Final colouring:", {v: solution[v] for v in VARS})
    ok = all(solution[a] != solution[b] for a, b in CONSTRAINTS)
    print("All constraints satisfied:", ok)
    return solution


# ---------------------------------------------------------------------------
# Question 4: Minimax and a heuristic evaluation function
# ---------------------------------------------------------------------------
TREE4 = [[[3, 5], [6, 9]], [[1, 2], [0, -1]]]


def minimax(node, is_max, name, values, best_child):
    if not isinstance(node, list):
        return node
    results = []
    for i, child in enumerate(node):
        results.append(minimax(child, not is_max, f"{name}{i + 1}", values, best_child))
    value = max(results) if is_max else min(results)
    values[name] = (("MAX" if is_max else "MIN"), value)
    best_child[name] = results.index(value)
    return value


def solve_minimax():
    print("=" * 60)
    print("QUESTION 4: MINIMAX")
    print("=" * 60)
    values, best = {}, {}
    root = minimax(TREE4, True, "R", values, best)
    for name in sorted(values, key=lambda n: (len(n), n)):
        kind, v = values[name]
        print(f"  node {name:<5} ({kind}) = {v}")
    print("Root value:", root, "| best first move: child", best["R"] + 1, "(left MIN node)")


def tictactoe_eval(board):
    """E = (# lines still open for X) - (# lines still open for O)."""
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]
    open_x = [l for l in lines if all(board[i] != "O" for i in l)]
    open_o = [l for l in lines if all(board[i] != "X" for i in l)]
    return open_x, open_o, len(open_x) - len(open_o)


def solve_tictactoe():
    # X | O |
    #   | X |
    #   |   | O
    board = ["X", "O", " ",
             " ", "X", " ",
             " ", " ", "O"]
    open_x, open_o, e = tictactoe_eval(board)
    print("Lines open for X:", len(open_x), open_x)
    print("Lines open for O:", len(open_o), open_o)
    print("E = ", len(open_x), "-", len(open_o), "=", e)


# ---------------------------------------------------------------------------
# Question 5: Alpha-beta pruning
# ---------------------------------------------------------------------------
LEAVES5 = [8, 6, 5, 7, 2, 9, 4, 3, 6, 1, 5, 2, 9, 7, 3, 4]


def build_tree(leaves):
    nodes = list(leaves)
    while len(nodes) > 1:
        nodes = [[nodes[i], nodes[i + 1]] for i in range(0, len(nodes), 2)]
    return nodes[0]


def count_leaves(node):
    return 1 if not isinstance(node, list) else sum(count_leaves(c) for c in node)


def alphabeta(node, depth, idx, alpha, beta, is_max, stats):
    name = f"{'ABCD'[depth]}{idx + 1}" if depth < 4 else None
    if not isinstance(node, list):
        stats["evaluated"] += 1
        return node
    print(f"  visit {name} ({'MAX' if is_max else 'MIN'})  alpha={alpha}, beta={beta}")
    value = float("-inf") if is_max else float("inf")
    for i, child in enumerate(node):
        v = alphabeta(child, depth + 1, idx * 2 + i, alpha, beta, not is_max, stats)
        if is_max:
            value = max(value, v)
            alpha = max(alpha, value)
        else:
            value = min(value, v)
            beta = min(beta, value)
        if alpha >= beta:
            rest = node[i + 1:]
            if rest:
                n_pruned = sum(count_leaves(c) for c in rest)
                stats["pruned"] += n_pruned
                print(f"  PRUNE after child {i + 1} of {name}: alpha={alpha} >= beta={beta}; "
                      f"skip {n_pruned} leaf/leaves")
            break
    print(f"  return {name} = {value}   (alpha={alpha}, beta={beta})")
    return value


def solve_alphabeta():
    print("=" * 60)
    print("QUESTION 5: ALPHA-BETA PRUNING")
    print("=" * 60)
    tree = build_tree(LEAVES5)
    stats = {"evaluated": 0, "pruned": 0}
    root = alphabeta(tree, 0, 0, float("-inf"), float("inf"), True, stats)
    print("Root value (alpha-beta):", root)
    print("Leaves evaluated:", stats["evaluated"], "| leaves pruned:", stats["pruned"])

    values, best = {}, {}
    mm_root = minimax(tree, True, "R", values, best)
    print("Root value (plain minimax):", mm_root, "| leaves evaluated by minimax:", len(LEAVES5))
    print("Same root value:", root == mm_root)


if __name__ == "__main__":
    greedy_best_first()
    a_star()
    check_admissibility()
    solve_csp()
    solve_minimax()
    solve_tictactoe()
    solve_alphabeta()
