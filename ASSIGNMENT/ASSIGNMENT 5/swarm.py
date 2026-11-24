"""Tasks A2 / B1: knowledge-grid simulator.  Grid cell = one document page to be extracted/OCR'd/embedded into the index.
blocked cell  = unreadable/corrupt page (obstacle);  restricted cell = confidential page (no-fly zone);  base = index server (0,0).
Energy = compute budget of a worker agent (battery); recharge = flush checkpoint and restart worker at base.
Modes: 'swarm' (decentralised node-counting + separation rule, N=1 is the single-agent baseline) and 'central' (controller assigns cells)."""
import numpy as np, time, math
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import shortest_path

R = C = 25; E0 = 200; RECHARGE = 8; R_COMM = 4; MARGIN = 4
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]       # N,E,S,W  (tie-break order used in the hand trace)

class World:
    def __init__(self, seed, R=R, C=C):
        rng = np.random.default_rng(1000 + seed); self.R, self.C = R, C
        g = np.zeros((R, C), int); r = rng.random((R, C)); g[r < 0.08] = 1; g[(r >= 0.08) & (r < 0.12)] = 2   # 1 blocked, 2 restricted
        g[0, 0] = 0; self.g = g
        free = (g == 0); idmap = -np.ones((R, C), int); cells = []
        # keep only cells reachable from base
        seen = {(0, 0)}; st = [(0, 0)]
        while st:
            a, b = st.pop()
            for da, db in DIRS:
                x, y = a + da, b + db
                if 0 <= x < R and 0 <= y < C and free[x, y] and (x, y) not in seen: seen.add((x, y)); st.append((x, y))
        for (a, b) in sorted(seen): idmap[a, b] = len(cells); cells.append((a, b))
        self.cells, self.idmap, self.n = cells, idmap, len(cells)
        A = lil_matrix((self.n, self.n))
        self.nbrs = []
        for i, (a, b) in enumerate(cells):
            nb = []
            for da, db in DIRS:
                x, y = a + da, b + db
                if 0 <= x < R and 0 <= y < C and idmap[x, y] >= 0: A[i, idmap[x, y]] = 1; nb.append(idmap[x, y])
                else: nb.append(-1)
            self.nbrs.append(nb)
        self.D, self.P = shortest_path(A.tocsr(), method="D", unweighted=True, return_predecessors=True)
        self.xy = np.array(cells)

def run(mode, N, fail_frac, seed, T=1000, ctrl_fail=False, link_loss=0.2, deterministic=False, starts=None, keep_trace=False):
    w = World(seed); rng = np.random.default_rng(seed * 7919 + N); n = w.n
    mark = np.zeros(n, int); mark[0] = 1
    pos = np.zeros(N, int) if starts is None else np.array(starts)
    for i in range(N): mark[pos[i]] += (0 if pos[i] == 0 else 1)
    energy = np.full(N, E0, float); alive = np.ones(N, bool); wait = np.zeros(N, int)
    k = int(math.floor(fail_frac * N)) + (rng.random() < fail_frac * N - math.floor(fail_frac * N))
    fail_t = np.full(N, 10**9); who = rng.choice(N, size=min(k, N), replace=False)
    fail_t[who] = rng.integers(int(0.05 * T), int(0.6 * T), size=len(who))
    target = -np.ones(N, int); ctrl_t = T // 3 if ctrl_fail else 10**9
    covered = 1; moves = revisit = 0; used = np.zeros(N); decs = 0; dec_ns = 0; msgs = 0; curve = []; t90 = None; trace = []
    xy = w.xy; need = int(math.ceil(0.95 * n))
    for t in range(T):
        alive &= fail_t > t
        if not alive.any(): curve.append(covered / n); continue
        if mode == "central" and t % 3 == 0 and t < ctrl_t:
            t0 = time.perf_counter_ns(); taken = set(target[target >= 0]); unc = np.where(mark == 0)[0]
            for i in rng.permutation(N):
                if not alive[i]: continue
                if target[i] >= 0 and mark[target[i]] == 0: continue
                cand = [u for u in unc if u not in taken]
                if not cand: target[i] = -1; continue
                cand = np.array(cand); j = cand[np.argmin(w.D[pos[i], cand])]
                if rng.random() > link_loss: target[i] = j; taken.add(j)       # assignment message delivered
            dec_ns += time.perf_counter_ns() - t0; decs += int(alive.sum()); msgs += 2 * int(alive.sum())
        for i in (np.arange(N) if deterministic else rng.permutation(N)):
            if not alive[i]: continue
            if wait[i] > 0:
                wait[i] -= 1
                if wait[i] == 0: energy[i] = E0
                continue
            u = pos[i]; t0 = time.perf_counter_ns(); nxt = u; layer = "deliberative"
            if energy[i] <= w.D[u, 0] + MARGIN and u != 0:                       # REACTIVE: battery low -> return to base
                nxt = w.P[0, u]; layer = "reactive-return"
            elif u == 0 and energy[i] < E0:
                wait[i] = RECHARGE; energy[i] = E0 if RECHARGE == 0 else energy[i]; continue
            elif mode == "central":
                tg = target[i]
                if tg >= 0 and tg != u and t < ctrl_t + 60: nxt = w.P[tg, u]
                else: continue                                                   # idle: waits for the controller
            else:
                others = [j for j in range(N) if j != i and alive[j]]
                known = [pos[j] for j in others if abs(xy[pos[j], 0] - xy[u, 0]) + abs(xy[pos[j], 1] - xy[u, 1]) <= R_COMM and rng.random() > 0.1 or False] if others else []
                occ = set(pos[j] for j in others)
                msgs += len(known)
                best = None
                for d, v in enumerate(w.nbrs[u]):
                    if v < 0 or v in occ: continue
                    dist = min((abs(xy[v, 0] - xy[o, 0]) + abs(xy[v, 1] - xy[o, 1]) for o in known), default=99)
                    key = (mark[v], -dist, d if deterministic else rng.random())
                    if best is None or key < best[0]: best = (key, v)
                nxt = best[1] if best else u
            dec_ns += time.perf_counter_ns() - t0; decs += 1
            if nxt != u:
                pos[i] = nxt; moves += 1; energy[i] -= 1; used[i] += 1
                if layer == "reactive-return": revisit += 1                      # transit only: no processing, no coverage
                elif mark[nxt] > 0: revisit += 1; mark[nxt] += 1
                else: covered += 1; energy[i] -= 1; used[i] += 1; mark[nxt] += 1
                if keep_trace: trace.append((t, int(i), tuple(xy[nxt]), layer))
            if energy[i] <= 0: alive[i] = False
        curve.append(covered / n)
        if t90 is None and covered >= need: t90 = t; break
    curve += [covered / n] * (T - len(curve))
    return dict(success=covered >= need, coverage=covered / n, t90=t90 if t90 is not None else T, curve=curve, redundancy=revisit / max(moves, 1),
                energy=float(used.sum()) / N, lat_us=dec_ns / 1000 / max(decs, 1), msgs_per_step=msgs / T, moves=moves, trace=trace, n_cells=n)

if __name__ == "__main__":
    import sys
    for mode, N in [("swarm", 1), ("swarm", 5), ("swarm", 25), ("central", 10)]:
        t0 = time.time(); r = run(mode, N, 0.0, 0)
        print(mode, N, "cov %.2f t90 %d red %.2f energy %.0f lat %.1fus" % (r["coverage"], r["t90"], r["redundancy"], r["energy"], r["lat_us"]), "time %.1fs" % (time.time() - t0), "cells", r["n_cells"])
