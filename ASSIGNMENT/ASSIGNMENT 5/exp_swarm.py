import swarm, numpy as np, json, time, pickle
T = 600; SEEDS = range(30); res = {}
t0 = time.time()
for N in [1, 5, 10, 25, 50]:
    for f in [0.0, 0.1, 0.3]:
        rs = [swarm.run("swarm", N, f, s, T=T) for s in SEEDS]; res[("swarm", N, f)] = rs
        print("swarm", N, f, "succ %.2f" % np.mean([r["success"] for r in rs]), "%.0fs" % (time.time() - t0), flush=True)
for N in [10, 25]:
    for f in [0.0, 0.3]:
        for cf in [False, True]:
            rs = [swarm.run("central", N, f, s, T=T, ctrl_fail=cf) for s in SEEDS]; res[("central" + ("-ctrlfail" if cf else ""), N, f)] = rs
            print("central", N, f, cf, "succ %.2f" % np.mean([r["success"] for r in rs]), "%.0fs" % (time.time() - t0), flush=True)
pickle.dump(res, open("out/swarm_results.pkl", "wb")); print("done")
