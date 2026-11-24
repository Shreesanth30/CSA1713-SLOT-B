import rdd_mini as R, time, json, numpy as np, pandas as pd
res = []; out = {}
for N in [10**5, 10**6, 10**7]:
    data = R.make_data(N)
    for P in [1, 2, 4, 8]:
        row = dict(N=N, P=P)
        for cache in [False, True]:
            walls, projs = [], []
            for rep in range(3):
                ctx = R.Ctx(); src, parsed, clean, per = R.build_pipeline(ctx, data, P, cache)
                t0 = time.perf_counter(); stats, top10 = R.job_stats(per); a = R.job_anomaly(clean); walls.append(time.perf_counter() - t0); projs.append(R.projected(ctx.tasks, P))
            row[f"wall_cache{int(cache)}"] = float(np.median(walls)); row[f"proj_cache{int(cache)}"] = float(np.median(projs)); row[f"shuffles_cache{int(cache)}"] = ctx.shuffles
            row[f"shuffle_records"] = ctx.shuffle_records; row["n_sensors"] = len(stats); row["anomalies"] = a
        # groupByKey variant (no map-side combine) at cache on
        ctx = R.Ctx(); src, parsed, clean, per = R.build_pipeline(ctx, data, P, True); g = clean.group_by_key_count(); t0 = time.perf_counter(); g.collect(); row["groupByKey_wall"] = time.perf_counter() - t0; row["groupByKey_records"] = ctx.shuffle_records
        res.append(row); print(row, flush=True)
    # correctness check against direct pandas
    df = pd.DataFrame({"s": data[0], "v": data[2]}); df = df[df.v > -900]
    # lineage recovery demo (P=8): lose one cached partition and recompute only it
    if N == 10**6:
        ctx = R.Ctx(); src, parsed, clean, per = R.build_pipeline(ctx, data, 8, True); R.job_anomaly(clean)
        t0 = time.perf_counter(); clean.compute(3); hit = time.perf_counter() - t0
        clean.lose_partition(3); t0 = time.perf_counter(); clean.compute(3); rec = time.perf_counter() - t0; t0 = time.perf_counter(); [clean.compute(i) for i in range(8)]; allc = time.perf_counter() - t0
        out["lineage"] = dict(cache_hit_ms=hit * 1000, recompute_one_partition_ms=rec * 1000, all_cached_ms=allc * 1000)
        full = R.Ctx(); s2, p2, c2, per2 = R.build_pipeline(full, data, 8, False); t0 = time.perf_counter(); [c2.compute(i) for i in range(8)]; out["lineage"]["recompute_all_ms"] = (time.perf_counter() - t0) * 1000
        # correctness
        ctx = R.Ctx(); s3, p3, c3, per3 = R.build_pipeline(ctx, data, 4, False); st, top = R.job_stats(per3)
        gain = (1 + 0.01 * (np.arange(5000) % 7)).astype(np.float32); off = (np.arange(5000) % 3).astype(np.float32); d2 = df.assign(v=df.v.values * gain[df.s.values] + off[df.s.values]); ref = d2.groupby("s").v.agg(["mean", "max"])
        m = st.set_index("s").join(ref, rsuffix="_ref"); out["correct"] = dict(max_abs_mean_err=float((m["mean"] - m["mean_ref"]).abs().max()), max_abs_max_err=float((m["max"] - m["max_ref"]).abs().max()), top10=[int(x) for x in top.s.values])
out["rows"] = res; json.dump(out, open("out/rdd_results.json", "w"), indent=1, default=float); print(out["lineage"], out["correct"])
