"""Tasks B3.1, B3.2, B3.4: full model vs compressed edge model for the assistant's intrusion detector."""
import numpy as np, json, time, zlib, pickle, sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, average_precision_score
from scipy.sparse import hstack, csr_matrix
from ids_data import generate, NUM, CLASSES
rows = generate(); n = len(rows)
tr = [r for r in rows if r["t"] < 0.6]; va = [r for r in rows if 0.6 <= r["t"] < 0.7]; te = [r for r in rows if r["t"] >= 0.7]
print(len(tr), len(va), len(te), flush=True)
LOGC = {"req_per_min", "failed_logins", "upload_kb", "resp_ms", "session_s", "query_len"}
def numf(r): return np.array([np.log1p(r[k]) if k in LOGC else r[k] for k in NUM], float)
Xn = {nm: np.array([numf(r) for r in d]) for nm, d in [("tr", tr), ("va", va), ("te", te)]}
y = {nm: np.array([r["y"] for r in d]) for nm, d in [("tr", tr), ("va", va), ("te", te)]}
sc = StandardScaler().fit(Xn["tr"]); Xs = {k: sc.transform(v) for k, v in Xn.items()}
def metrics(yt, proba):
    yp = proba.argmax(1); p, r, f, _ = precision_recall_fscore_support(yt, yp, labels=range(6), zero_division=0)
    Y = np.eye(6)[yt]
    return dict(accuracy=accuracy_score(yt, yp), macro_precision=p.mean(), macro_recall=r.mean(), macro_f1=f.mean(), per_class_f1=f.tolist(), per_class_precision=p.tolist(), per_class_recall=r.tolist(),
                roc_auc=roc_auc_score(Y, proba, average="macro"), pr_auc=average_precision_score(Y, proba, average="macro"), confusion=confusion_matrix(yt, yp, labels=range(6)).tolist())
# ---------------- FULL MODEL: char 2-5 gram TF-IDF (20k) + numerics -> MLP(256,128) ---------------
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), max_features=20000, sublinear_tf=True).fit([r["text"] for r in tr])
def fullX(d, Xsd): return hstack([vec.transform([r["text"] for r in d]), csr_matrix(Xsd)]).tocsr()
t0 = time.time(); full = MLPClassifier((256, 128), max_iter=12, batch_size=256, random_state=0, alpha=1e-4).fit(fullX(tr, Xs["tr"]), y["tr"]); print("full trained %.0fs" % (time.time() - t0), flush=True)
Pf = full.predict_proba(fullX(te, Xs["te"])); mf = metrics(y["te"], Pf)
params_full = sum(w.size for w in full.coefs_) + sum(b.size for b in full.intercepts_)
macs_dense = sum(w.shape[0] * w.shape[1] for w in full.coefs_)
print("FULL", {k: round(v, 4) for k, v in mf.items() if isinstance(v, float)}, "params", params_full, flush=True)
# ---------------- COMPRESSED: hashed 3-4 grams (1024) + numerics -> MLP(32,16) -> int8; cascade with LR stage 0 -------------
H = 1024
def hfeat(text):
    t = text.lower(); idx = {}
    for k in (3, 4):
        for i in range(len(t) - k + 1): j = zlib.crc32(t[i:i + k].encode()) & (H - 1); idx[j] = idx.get(j, 0) + 1
    ii = np.fromiter(idx.keys(), int, len(idx)); vv = np.log1p(np.fromiter(idx.values(), float, len(idx)))
    nrm = np.sqrt((vv * vv).sum()) or 1.0
    return ii, (vv / nrm).astype(np.float32)
def denseH(d):
    M = np.zeros((len(d), H), np.float32)
    for i, r in enumerate(d): ii, vv = hfeat(r["text"]); M[i, ii] = vv
    return M
Hm = {nm: denseH(d) for nm, d in [("tr", tr), ("va", va), ("te", te)]}
small = MLPClassifier((32, 16), max_iter=60, random_state=0, alpha=1e-4).fit(np.hstack([Hm["tr"], Xs["tr"]]), y["tr"])
W = [w.astype(np.float32) for w in small.coefs_]; B = [b.astype(np.float32) for b in small.intercepts_]
def q8(w): s = np.abs(w).max() / 127.0; return np.round(w / s).astype(np.int8), np.float32(s)
Q = [q8(w) for w in W]
def softmax(z): e = np.exp(z - z.max()); return e / e.sum()
def infer_small(ii, vv, xn, quant):
    if quant:
        (q1, s1), (q2, s2), (q3, s3) = Q
        h = np.maximum(q1[:H][ii].astype(np.float32).T @ vv * s1 + xn @ (q1[H:].astype(np.float32) * s1) + B[0], 0)
        h = np.maximum(h @ (q2.astype(np.float32) * s2) + B[1], 0); return softmax(h @ (q3.astype(np.float32) * s3) + B[2])
    h = np.maximum(W[0][:H][ii].T @ vv + xn @ W[0][H:] + B[0], 0); h = np.maximum(h @ W[1] + B[1], 0); return softmax(h @ W[2] + B[2])
def predict_small(d, Xsd, quant): return np.array([infer_small(*hfeat(r["text"]), Xsd[i].astype(np.float32), quant) for i, r in enumerate(d)])
Ps = predict_small(te, Xs["te"], False); Pq = predict_small(te, Xs["te"], True); ms = metrics(y["te"], Ps); mq = metrics(y["te"], Pq)
lr0 = LogisticRegression(max_iter=500, C=2.0).fit(Xs["tr"], y["tr"])        # stage 0: numerics only (14 features)
p0v = lr0.predict_proba(Xs["va"]); 
THETA = None
for th in [0.999, 0.995, 0.99, 0.98, 0.95]:          # choose the lowest exit threshold whose VALIDATION exit-accuracy is >= 99.5%
    ex = p0v.max(1) >= th
    if ex.sum() and (p0v.argmax(1)[ex] == y["va"][ex]).mean() >= 0.995: THETA = th
THETA = THETA or 0.999; print("cascade theta", THETA, flush=True)
def predict_cascade(d, Xsd):
    P = np.zeros((len(d), 6)); exited = 0
    p0 = lr0.predict_proba(Xsd)
    for i, r in enumerate(d):
        if p0[i].max() >= THETA: P[i] = p0[i]; exited += 1
        else: P[i] = infer_small(*hfeat(r["text"]), Xsd[i].astype(np.float32), True)
    return P, exited / len(d)
Pc, exit_rate = predict_cascade(te, Xs["te"]); mc = metrics(y["te"], Pc)
for nm, m in [("small fp32", ms), ("small int8", mq), ("cascade", mc)]: print(nm, {k: round(v, 4) for k, v in m.items() if isinstance(v, float)}, flush=True)
# ---------------- latency per sample (single thread, end-to-end: raw record -> class) ----------------
sub = te[:1500]; Xsub = Xs["te"][:1500]
def timeit(fn):
    ts = []
    for i in range(len(sub)):
        t0 = time.perf_counter_ns(); fn(i); ts.append(time.perf_counter_ns() - t0)
    ts = np.array(ts) / 1e6; return dict(mean_ms=float(ts.mean()), p95_ms=float(np.percentile(ts, 95)), throughput_per_s=float(1000 / ts.mean()))
def f_full(i): r = sub[i]; x = hstack([vec.transform([r["text"]]), csr_matrix(sc.transform(numf(r)[None]))]).tocsr(); full.predict_proba(x)
def f_small(i): r = sub[i]; xn = ((numf(r) - sc.mean_) / sc.scale_).astype(np.float32); infer_small(*hfeat(r["text"]), xn, False)
def f_q8(i): r = sub[i]; xn = ((numf(r) - sc.mean_) / sc.scale_).astype(np.float32); infer_small(*hfeat(r["text"]), xn, True)
def f_casc(i):
    r = sub[i]; xn = ((numf(r) - sc.mean_) / sc.scale_); p0 = lr0.predict_proba(xn[None])[0] if False else None
    z = xn @ lr0.coef_.T + lr0.intercept_; p = softmax(z)
    if p.max() < THETA: infer_small(*hfeat(r["text"]), xn.astype(np.float32), True)
lat = dict(full_sklearn=timeit(f_full), small_fp32=timeit(f_small), small_int8=timeit(f_q8), cascade_int8=timeit(f_casc))
print(json.dumps(lat, indent=1), flush=True)
import pickle as pk, io
full._optimizer = None; vec.stop_words_ = None; vocab_n = len(vec.vocabulary_); size_full = len(pk.dumps((vec, sc, full))); params_small = sum(w.size for w in W) + sum(b.size for b in B)
size = dict(vocab_size=vocab_n, full_weights_fp32_bytes=int(params_full*4), full_bytes=size_full, small_fp32_bytes=int(params_small * 4), small_int8_bytes=int(sum(q.size for q, _ in Q) + sum(b.size * 4 for b in B) + lr0.coef_.size * 4 + 64), params_full=int(params_full), params_small=int(params_small), macs_dense_full=int(macs_dense),
            macs_small_sparse=int(70 * 32 + 14 * 32 + 32 * 16 + 16 * 6))
# ---------------- B3.2 error analysis on the full model ----------------
yp = Pf.argmax(1); wrong = np.where(yp != y["te"])[0]
var_rec = {}
for v in ["known", "low_slow", "obfuscated", "polyglot"]:
    idx = [i for i, r in enumerate(te) if r["variant"] == v and r["y"] != 0]
    var_rec[v] = dict(n=len(idx), recall_full=float(np.mean([yp[i] == y["te"][i] for i in idx])) if idx else None, recall_cascade=float(np.mean([Pc[i].argmax() == y["te"][i] for i in idx])) if idx else None)
def cat(i):
    r = te[i]
    if r["variant"] != "known": return "unseen attack variant: " + r["variant"]
    if r["y"] == 0: return "false alarm on normal traffic"
    if CLASSES[yp[i]] == "normal": return "missed known attack (looks normal)"
    return "confused between attack types"
errs = [dict(idx=int(i), t=round(te[i]["t"], 3), true=CLASSES[te[i]["y"]], pred=CLASSES[yp[i]], category=cat(i), text=te[i]["text"][:70], failed=int(te[i]["failed_logins"]), users=int(te[i]["distinct_usernames"]),
             rpm=round(te[i]["req_per_min"], 1), upload=round(te[i]["upload_kb"]), ext_risk=te[i]["ext_risk"]) for i in wrong]
from collections import Counter
cats = Counter(e["category"] for e in errs)
# pick one example of each category (>=5 mispredictions)
shown = []; seen = Counter()
for e in errs:
    if seen[e["category"]] < 2: shown.append(e); seen[e["category"]] += 1
from ids_data import INJ_NEW
def subtype(r):
    if r["variant"] == "polyglot": return "polyglot upload (clean extension)"
    if r["variant"] == "low_slow": return "low-and-slow credential spraying"
    for k, nm in zip(INJ_NEW, ["leetspeak obfuscation", "paraphrase / synonym", "role-play social engineering", "base64-encoded payload"]):
        if k in r["text"]: return nm
    return "other"
sub_rec = {}
for st in sorted({subtype(r) for r in te if r["variant"] != "known"}):
    idx = [i for i, r in enumerate(te) if r["variant"] != "known" and subtype(r) == st]
    sub_rec[st] = dict(n=len(idx), recall_full=float(np.mean([yp[i] == y["te"][i] for i in idx])), recall_cascade=float(np.mean([Pc[i].argmax() == y["te"][i] for i in idx])))
err_sub = Counter(subtype(te[i]) for i in wrong)
shown2 = []; seen2 = Counter()
for i in wrong:
    st = subtype(te[i])
    if seen2[st] < 2: shown2.append(dict(idx=int(i), true=CLASSES[y["te"][i]], pred=CLASSES[yp[i]], category=st, text=te[i]["text"][:75])); seen2[st] += 1
# stress: sensor/feature noise robustness (gaussian noise on numeric features)
rng = np.random.default_rng(0); noise = {}
for s_ in [0.0, 0.5, 1.0, 2.0]:
    Xn_ = Xs["te"] + rng.normal(0, s_, Xs["te"].shape); noise[str(s_)] = dict(full=float(accuracy_score(y["te"], full.predict(fullX(te, Xn_)))), small_int8=float(accuracy_score(y["te"], predict_small(te, Xn_, True).argmax(1))))
res = dict(n_train=len(tr), n_val=len(va), n_test=len(te), full=mf, small_fp32=ms, small_int8=mq, cascade=mc, exit_rate=exit_rate, theta=THETA, latency=lat, size=size, variant_recall=var_rec, error_categories=dict(cats), n_errors=len(errs), shown=shown, noise=noise, sub_rec=sub_rec, err_sub=dict(err_sub), shown2=shown2,
           class_counts_test=np.bincount(y["te"], minlength=6).tolist())
json.dump(res, open("out/ids_results.json", "w"), indent=1)
# save test predictions of the compressed cascade + features for the chatbot (B4)
pickle.dump(dict(te=te, pred_cascade=Pc.argmax(1), proba=Pc, pred_full=Pf.argmax(1), W=W, B=B), open("out/ids_test_preds.pkl", "wb"))
print("errors", len(errs), dict(cats)); print(var_rec); print(size); print(noise)

pickle.dump((vec, sc, full), open("out/full_model.pkl", "wb"))
pickle.dump(dict(Q=Q, B=B, lr_w=lr0.coef_.astype(np.float32), lr_b=lr0.intercept_.astype(np.float32), mean=sc.mean_.astype(np.float32), scale=sc.scale_.astype(np.float32), theta=THETA), open("out/edge_model.pkl", "wb"))
import os; print("file sizes", os.path.getsize("out/full_model.pkl"), os.path.getsize("out/edge_model.pkl"))
