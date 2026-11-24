import pickle, json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": .3, "figure.dpi": 130})
T = 600; res = pickle.load(open("out/swarm_results.pkl", "rb")); tab = {}
def agg(rs):
    ci = lambda x: 1.96 * np.std(x, ddof=1) / np.sqrt(len(x))
    t95 = [r["t90"] for r in rs]
    return dict(success=float(np.mean([r["success"] for r in rs])), coverage=float(np.mean([r["coverage"] for r in rs])), t95=float(np.mean(t95)), t95_ci=float(ci(t95)), redundancy=float(np.mean([r["redundancy"] for r in rs])),
                energy=float(np.mean([r["energy"] for r in rs])), lat_us=float(np.mean([r["lat_us"] for r in rs])), msgs=float(np.mean([r["msgs_per_step"] * T / max(r["t90"], 1) for r in rs])), cov_ci=float(ci([r["coverage"] for r in rs])))
rows = []
for (mode, N, f), rs in sorted(res.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
    a = agg(rs); a.update(mode=mode, N=N, f=f); a['energy_total'] = a['energy'] * N; rows.append(a)
tab["swarm"] = rows
for r in rows: print(r["mode"], r["N"], r["f"], "succ %.2f cov %.3f t95 %.0f±%.0f red %.2f energy %.0f lat %.1f msgs %.1f" % (r["success"], r["coverage"], r["t95"], r["t95_ci"], r["redundancy"], r["energy"], r["lat_us"], r["msgs"]))
# ---- fig 1: coverage vs time
fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
for a, f in zip(ax, [0.0, 0.1, 0.3]):
    for N in [1, 5, 10, 25, 50]:
        c = np.mean([r["curve"] for r in res[("swarm", N, f)]], axis=0) * 100; a.plot(np.arange(T), c, label=f"swarm N={N}" if N > 1 else "single drone")
    a.axhline(95, color="k", ls=":", lw=.8); a.set_title(f"{int(f*100)}% of agents fail"); a.set_xlabel("time step")
ax[0].set_ylabel("coverage of coverable pages (%)"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("out/fig_coverage.png"); plt.close()
# ---- fig 2: scaling metrics
fig, ax = plt.subplots(1, 4, figsize=(11, 2.9)); Ns = [1, 5, 10, 25, 50]
g = lambda N, f, k: [r[k] for r in rows if r["mode"] == "swarm" and r["N"] == N and r["f"] == f][0]
for f, ls in [(0.0, "-o"), (0.3, "--s")]:
    ax[0].plot(Ns, [g(N, f, "t95") for N in Ns], ls, label=f"{int(f*100)}% fail"); ax[1].plot(Ns, [g(N, f, "redundancy") for N in Ns], ls); ax[2].plot(Ns, [g(N, f, "energy") for N in Ns], ls); ax[3].plot(Ns, [g(N, f, "lat_us") for N in Ns], ls)
for a, t in zip(ax, ["steps to 95% coverage (cap 600)", "redundancy ratio", "energy used per agent", "decision latency (µs)"]): a.set_title(t, fontsize=8); a.set_xscale("log"); a.set_xlabel("N agents"); a.set_xticks(Ns); a.set_xticklabels(Ns)
ax[0].legend(); plt.tight_layout(); plt.savefig("out/fig_scaling.png"); plt.close()
# ---- fig 3: central vs swarm
fig, ax = plt.subplots(1, 2, figsize=(8, 3.1), sharey=True)
for a, N in zip(ax, [10, 25]):
    for key, lab in [(("swarm", N, 0.0), "swarm"), (("central", N, 0.0), "central, links 20% lossy"), (("central-ctrlfail", N, 0.0), "central, controller dies at t=200")]:
        a.plot(np.mean([r["curve"] for r in res[key]], axis=0) * 100, label=lab)
    a.set_title(f"N = {N}, no agent failures"); a.set_xlabel("time step")
ax[0].set_ylabel("coverage (%)"); ax[0].legend(fontsize=7); plt.tight_layout(); plt.savefig("out/fig_central.png"); plt.close()
# ---- RDD
R = json.load(open("out/rdd_results.json")); fig, ax = plt.subplots(1, 3, figsize=(10, 2.9))
for a, N in zip(ax, [10**5, 10**6, 10**7]):
    rr = [r for r in R["rows"] if r["N"] == N]; P = [r["P"] for r in rr]
    a.plot(P, [r["wall_cache1"] for r in rr], "-o", label="measured wall (1 core), cache"); a.plot(P, [r["wall_cache0"] for r in rr], "--o", label="measured wall (1 core), no cache")
    a.plot(P, [r["proj_cache1"] for r in rr], "-s", label="projected P cores, cache"); a.set_title(f"N = {N:,}"); a.set_xlabel("partitions"); a.set_xticks(P); a.set_ylabel("seconds")
ax[0].legend(fontsize=6); plt.tight_layout(); plt.savefig("out/fig_rdd.png"); plt.close()
# ---- IDS
I = json.load(open("out/ids_results.json")); CL = ["normal", "brute\nforce", "prompt\ninj.", "dos\nflood", "mal.\nupload", "scraping"]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.8))
for a, key, t in zip(ax, ["full", "cascade"], ["Full MLP", "Compressed cascade (int8)"]):
    cm = np.array(I[key]["confusion"]); a.imshow(np.log1p(cm), cmap="Blues"); a.set_title(t); a.grid(False)
    a.set_xticks(range(6)); a.set_xticklabels(CL, fontsize=6); a.set_yticks(range(6)); a.set_yticklabels(CL, fontsize=6); a.set_xlabel("predicted"); a.set_ylabel("true")
    for i in range(6):
        for j in range(6): a.text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="k" if cm[i, j] < cm.max() * .5 else "w")
plt.tight_layout(); plt.savefig("out/fig_confusion.png"); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(9, 3))
vr = I["sub_rec"]; names = list(vr); x = np.arange(len(names)); ax[0].bar(x - .2, [vr[n]["recall_full"] for n in names], .4, label="full"); ax[0].bar(x + .2, [vr[n]["recall_cascade"] for n in names], .4, label="cascade")
ax[0].set_xticks(x); ax[0].set_xticklabels([n.replace(" / ", "/\n").replace(" ", "\n", 1) for n in names], fontsize=6); ax[0].set_title("Recall on attack variants unseen in training"); ax[0].legend(fontsize=7)
ns = list(I["noise"]); ax[1].plot([float(n) for n in ns], [I["noise"][n]["full"] for n in ns], "-o", label="full"); ax[1].plot([float(n) for n in ns], [I["noise"][n]["small_int8"] for n in ns], "-s", label="compressed int8")
ax[1].set_xlabel("Gaussian noise σ added to standardised numeric features"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Robustness to sensor/log noise"); ax[1].legend(fontsize=7)
plt.tight_layout(); plt.savefig("out/fig_ids_fail.png"); plt.close()
# ---- chatbot
Cb = json.load(open("out/chatbot_results.json")); fig, ax = plt.subplots(1, 3, figsize=(11, 3))
w = list(Cb["wer"]); ax[0].plot([float(x) * 100 for x in w], [Cb["wer"][x]["intent_accuracy"] for x in w], "-o", label="in-scope intent accuracy"); ax[0].plot([float(x) * 100 for x in w], [Cb["wer"][x]["oos_detection"] for x in w], "-s", label="OOS detection")
ax[0].plot([float(x) * 100 for x in w], [Cb["wer"][x]["false_accept_rate"] for x in w], "-^", label="false-accept rate"); ax[0].set_xlabel("word error rate (%)"); ax[0].legend(fontsize=6); ax[0].set_title("Degradation under ASR errors")
taus = Cb["taus"]
for x in w:
    sw = Cb["sweep"][x]; ax[1].plot(taus, [sw[str(t)][0] for t in taus], label=f"WER {int(float(x)*100)}%"); ax[2].plot(taus, [sw[str(t)][1] for t in taus], label=f"WER {int(float(x)*100)}%")
ax[1].set_title("Threshold sweep: in-scope accuracy (accepted & correct)"); ax[2].set_title("Threshold sweep: false-accept rate (OOS accepted)"); 
for a in ax[1:]: a.set_xlabel("confidence threshold τ"); a.axvline(Cb["tau"], color="k", ls=":", lw=.8)
ax[1].legend(fontsize=6); plt.tight_layout(); plt.savefig("out/fig_chat.png"); plt.close()
# latency/size figure
L = I["latency"]; fig, ax = plt.subplots(1, 2, figsize=(8, 2.9)); nm = list(L); ax[0].bar(range(4), [L[n]["mean_ms"] for n in nm], .4, label="mean"); ax[0].bar(np.arange(4) + .4, [L[n]["p95_ms"] for n in nm], .4, label="p95")
ax[0].set_yscale("log"); ax[0].set_xticks(np.arange(4) + .2); ax[0].set_xticklabels([n.replace("_", "\n") for n in nm], fontsize=7); ax[0].set_ylabel("ms / sample (sandbox)"); ax[0].legend(fontsize=7); ax[0].set_title("Latency per sample")
sz = I["size"]; vals = [sz["full_bytes"] / 1e6, sz["small_fp32_bytes"] / 1e6, sz["small_int8_bytes"] / 1e6]; ax[1].bar(["full\n(pickle)", "small\nfp32", "small\nint8"], vals); ax[1].set_yscale("log"); ax[1].set_ylabel("MB"); ax[1].set_title("Model size")
plt.tight_layout(); plt.savefig("out/fig_latency.png"); plt.close()
json.dump(tab, open("out/swarm_table.json", "w"), indent=1)
print("noisy sensors true:", sorted(np.random.default_rng(1).choice(5000, 10, replace=False)) if False else "")
