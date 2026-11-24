"""Deterministic 5x5 version of the A2 rule, used ONLY to cross-check the hand traces of A2.3 / A2.4 (Task B1.2)."""
D = [(-1, 0), (0, 1), (1, 0), (0, -1)]
def step(pos, m, n=5):
    for i in range(len(pos)):
        r, c = pos[i]; others = [p for j, p in enumerate(pos) if j != i]; best = None
        for d, (dr, dc) in enumerate(D):
            v = (r + dr, c + dc)
            if not (0 <= v[0] < n and 0 <= v[1] < n) or v in others: continue
            dist = min(abs(v[0]-o[0]) + abs(v[1]-o[1]) for o in others)
            key = (m[v], -dist, d)
            if best is None or key < best[0]: best = (key, v)
        if best: pos[i] = best[1]; m[best[1]] += 1
    return pos
def trace(pos, m, steps):
    pos = list(pos); out = []
    for t in range(1, steps + 1):
        pos = step(pos, m); out.append((t, list(pos), sum(1 for v in m.values() if v > 0)))
    return out
if __name__ == "__main__":
    m = {(r, c): 0 for r in range(5) for c in range(5)}
    start = [(0, 0), (0, 4), (4, 2)]
    for p in start: m[p] += 1
    print("A2.3"); [print(x) for x in trace(start, m, 6)]
    m = {(r, c): 1 for r in range(5) for c in range(5)}
    for p in [(3, 0), (3, 1), (4, 0), (4, 1)]: m[p] = 0
    print("A2.4"); moves = 0
    start = [(0, 4), (0, 3), (1, 4)]; pos = list(start)
    for t in range(1, 7):
        before = list(pos); pos = step(pos, m); moves += sum(a != b for a, b in zip(before, pos))
        print(t, pos, "unmarked left", sum(1 for v in m.values() if v == 0))
    print("moves", moves)
