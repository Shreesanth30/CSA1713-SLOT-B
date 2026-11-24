"""Minimal RDD engine used for Task B3.3 because PySpark could not be installed in the offline sandbox.
Faithful to Spark semantics: lazy transformations build lineage; actions trigger jobs; narrow deps are pipelined per partition;
reduceByKey = wide dependency (map-side combine + shuffle); cache() keeps computed partitions; a lost partition is recomputed from lineage.
Execution is sequential (the sandbox has 1 CPU core); per-task times are recorded so a P-core makespan can be PROJECTED."""
import numpy as np, pandas as pd, time

class Ctx:
    def __init__(self): self.shuffles = 0; self.shuffle_records = 0; self.tasks = []; self.task_overhead = 0.005   # 5 ms per task (assumed Spark scheduling overhead)
class RDD:
    def __init__(self, ctx, nparts, fn, parent=None, name="", wide=False):
        self.ctx, self.n, self.fn, self.parent, self.name, self.wide = ctx, nparts, fn, parent, name, wide
        self.cached = False; self.store = {}; self.shuffle_out = None
    # ---- transformations (lazy) ----
    def map_partitions(self, fn, name="map"): return RDD(self.ctx, self.n, fn, self, name)
    def cache(self): self.cached = True; return self
    def reduce_by_key(self, nout=None, name="reduceByKey"):
        r = RDD(self.ctx, nout or self.n, None, self, name, wide=True); return r
    def map_values(self, fn, name="mapValues"): return RDD(self.ctx, self.n, lambda df: fn(df), self, name)
    def group_by_key_count(self): r = RDD(self.ctx, self.n, None, self, "groupByKey", wide=True); r.group = True; return r
    # ---- computation ----
    def compute(self, i):
        if self.cached and i in self.store: return self.store[i]
        t0 = time.perf_counter()
        if self.wide:
            if self.shuffle_out is None: self._shuffle_write()
            data = self._shuffle_read(i)
        else:
            inp = self.parent.compute(i) if self.parent is not None else None
            data = self.fn(i, inp) if self.parent is None else self.fn(inp)
        self.ctx.tasks.append((self.name, i, time.perf_counter() - t0))
        if self.cached: self.store[i] = data
        return data
    def _shuffle_write(self):                                    # map side: every parent partition is computed once, combined locally, hash-partitioned
        self.ctx.shuffles += 1; outs = [[] for _ in range(self.n)]
        for p in range(self.parent.n):
            t0 = time.perf_counter(); df = self.parent.compute(p)
            if getattr(self, "group", False): comb = df                       # groupByKey: ALL records are shuffled
            else:
                df = df.assign(v2=df.v.astype(np.float64) ** 2); g = df.groupby("s", sort=False)
                comb = pd.DataFrame({"n": g.v.count(), "sum": g.v.sum(), "sumsq": g.v2.sum(), "max": g.v.max()}).reset_index()   # map-side combine
            self.ctx.shuffle_records += len(comb)
            h = (comb["s"].values % self.n)
            for o in range(self.n): outs[o].append(comb[h == o])
            self.ctx.tasks.append(("shuffle-write", p, time.perf_counter() - t0))
        self.shuffle_out = outs
    def _shuffle_read(self, i):
        df = pd.concat(self.shuffle_out[i])
        if getattr(self, "group", False):
            g = df.groupby("s", sort=False).v; return pd.DataFrame({"n": g.count(), "sum": g.sum(), "max": g.max()}).reset_index()
        g = df.groupby("s", sort=False); out = g[["n", "sum", "sumsq"]].sum(); out["max"] = g["max"].max(); return out.reset_index()
    # ---- actions (trigger a job) ----
    def collect(self): return pd.concat([self.compute(i) for i in range(self.n)])
    def count(self): return sum(len(self.compute(i)) if not isinstance(self.compute(i), (int, np.integer)) else self.compute(i) for i in range(self.n))
    def lose_partition(self, i): self.store.pop(i, None)

def make_data(N, K=5000, seed=1):
    rng = np.random.default_rng(seed); p = 1.0 / np.arange(1, K + 1) ** 0.25; p /= p.sum()
    s = np.searchsorted(np.cumsum(p), rng.random(N)).astype(np.int32); sigma = np.full(K, 1.0); sigma[rng.choice(K, 10, replace=False)] = 8.0
    v = (rng.normal(0, 1, N) * sigma[s] + (s % 50)).astype(np.float32); bad = rng.random(N) < 0.01; v[bad] = -999.0
    return s, np.arange(N, dtype=np.int32), v
GAIN = None
def build_pipeline(ctx, data, P, cache_clean):
    s, ts, v = data; N = len(s); K = int(s.max()) + 1; gain = (1 + 0.01 * (np.arange(K) % 7)).astype(np.float32); off = (np.arange(K) % 3).astype(np.float32)
    bounds = np.linspace(0, N, P + 1).astype(int)
    src = RDD(ctx, P, lambda i, _: (s[bounds[i]:bounds[i + 1]], ts[bounds[i]:bounds[i + 1]], v[bounds[i]:bounds[i + 1]]), None, "textFile")
    parsed = src.map_partitions(lambda t: pd.DataFrame({"s": t[0], "ts": t[1], "v": t[2]}), "parse")                       # narrow
    clean = parsed.map_partitions(lambda df: df[df.v > -900].assign(v=lambda d: (d.v.values * gain[d.s.values] + off[d.s.values]).astype(np.float32)), "filter+calibrate")   # narrow
    if cache_clean: clean.cache()
    per = clean.reduce_by_key(name="reduceByKey")                                                                           # wide: ONE shuffle
    return src, parsed, clean, per
def job_stats(per):
    t = per.collect(); t["mean"] = t["sum"] / t["n"]; t["var"] = t["sumsq"] / t["n"] - t["mean"] ** 2
    return t[["s", "mean", "max", "var"]], t.nlargest(10, "var")                                                          # collect + takeOrdered(10)
def job_anomaly(clean, thr=40.0): return sum(int((clean.compute(i).v > thr).sum()) for i in range(clean.n))                    # 2nd action re-using `clean`
def projected(tasks, P, ctx_overhead=0.005):
    """makespan if each of P partitions ran on its own core: per stage the slowest task + scheduling overhead; stages are sequential"""
    by = {}
    for name, i, dt in tasks: by.setdefault(name, []).append(dt)
    return sum(max(v) + ctx_overhead for v in by.values())
