"""Tasks A3 / B2: generic expert-system shell + RAG-pipeline fault-diagnosis rule base.
Shell components: KnowledgeBase (rules as DATA), working memory (dict), InferenceEngine (forward chaining + conflict resolution),
Explanation facility (HOW / WHY), Knowledge-acquisition interface (add/edit rule from a plain dict / JSON, no engine change)."""
import json, copy, operator

OPS = {"==": operator.eq, "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge}

class Rule:
    def __init__(self, rid, conds, then, prio, seq, text=""):
        self.id, self.conds, self.then, self.prio, self.seq, self.text = rid, conds, then, prio, seq, text
    def matches(self, wm):
        for a, op, v in self.conds:
            if a not in wm or not OPS[op](wm[a], v): return False      # unknown evidence never satisfies a condition
        return True
    def spec(self): return len(self.conds)
    def to_dict(self): return dict(id=self.id, if_=self.conds, then=self.then, prio=self.prio, text=self.text)

class Shell:
    def __init__(self, rules): self.rules = rules; self.next_seq = max(r.seq for r in rules) + 1
    # ---- knowledge acquisition: rules enter as data ---------------------------------
    def add_rule(self, d):
        self.rules.append(Rule(d["id"], [tuple(c) for c in d["if"]], tuple(d["then"]), d["prio"], self.next_seq, d.get("text", ""))); self.next_seq += 1
    def edit_rule(self, rid, **kw):
        r = next(r for r in self.rules if r.id == rid)
        if "if" in kw: r.conds = [tuple(c) for c in kw["if"]]
        if "prio" in kw: r.prio = kw["prio"]
        if "then" in kw: r.then = tuple(kw["then"])
    # ---- inference cycle ----------------------------------------------------------------
    def run(self, facts, strategy="specificity"):
        wm = dict(facts); fired = []; why_asked = []
        changed = True
        while changed:                                                  # 1) derivation rules (hypotheses) fire to quiescence
            changed = False
            for r in self.rules:
                if r.then[0] == "hyp" and r.id not in fired and r.matches(wm):
                    wm[r.then[1]] = True; fired.append(r.id); changed = True
        conflict = [r for r in self.rules if r.then[0] == "diagnosis" and r.matches(wm)]       # 2) conflict set
        if not conflict: return dict(diagnosis="UNKNOWN", fired=fired, conflict=[], wm=wm)
        pos = {r.id: i for i, r in enumerate(self.rules)}
        keyf = {"specificity": lambda r: (-r.spec(), -r.prio, pos[r.id]),            # 3) conflict resolution
                "priority": lambda r: (-r.prio, pos[r.id]),
                "recency": lambda r: (-r.seq, pos[r.id]),
                "order": lambda r: (pos[r.id],)}[strategy]
        win = sorted(conflict, key=keyf)[0]; fired.append(win.id)                  # 4) act (single-fault assumption: stop)
        return dict(diagnosis=win.then[1], fired=fired, conflict=[r.id for r in conflict], winner=win.id, wm=wm)
    # ---- explanation facility ---------------------------------------------------------------
    def how(self, res):
        if res["diagnosis"] == "UNKNOWN": return "HOW: no rule's conditions were satisfied by the known evidence."
        out = []
        for rid in res["fired"]:
            r = next(r for r in self.rules if r.id == rid)
            out.append(f'{rid}: IF ' + " AND ".join(f"{a} {op} {v}" for a, op, v in r.conds) + f' THEN {r.then[1]}  [{r.text}]')
        return f'HOW {res["diagnosis"]}: ' + " ; ".join(out) + (f' (competing rules: {sorted(set(res["conflict"])-{res["winner"]})})' if len(res["conflict"]) > 1 else "")
    def why(self, attr):
        rs = [r.id for r in self.rules if any(c[0] == attr for c in r.conds)]
        return f"WHY ask '{attr}'? Rules {rs} need it to decide between candidate faults."

def R(rid, conds, then, prio, seq, text): return Rule(rid, conds, then, prio, seq, text)
def build_rules():
    return [
     R("H1", [("scanned_source", "==", True), ("ocr_conf", "<", 0.6)], ("hyp", "ocr_suspect"), 0, 1, "scanned page with low OCR confidence"),
     R("H2", [("gold_doc_in_topk", "==", False)], ("hyp", "retrieval_gap"), 0, 2, "known answer document missing from top-k"),
     R("H3", [("chunk_len", "<", 40)], ("hyp", "chunk_size_abnormal"), 0, 3, "chunk shorter than 40 tokens"),
     R("H4", [("chunk_len", ">", 400)], ("hyp", "chunk_size_abnormal"), 0, 4, "chunk longer than 400 tokens"),
     R("H5", [("newer_version_exists", "==", True), ("answer_matches_old_value", "==", True)], ("hyp", "outdated_evidence"), 0, 5, "answer equals value of superseded document"),
     R("R1", [("scanned_source", "==", True), ("garbled_text", "==", True)], ("diagnosis", "OCR_FAILURE"), 9, 6, "scanned page + garbled text"),
     R("R2", [("scanned_source", "==", True), ("digits_wrong", "==", True)], ("diagnosis", "OCR_FAILURE"), 9, 7, "scanned page + misread digits"),
     R("R3", [("ocr_suspect", "==", True)], ("diagnosis", "OCR_FAILURE"), 6, 8, "low OCR confidence"),
     R("R4", [("key_fact_split", "==", True), ("answer_truncated", "==", True)], ("diagnosis", "CHUNKING_FAULT"), 9, 9, "fact split over chunks and answer cut"),
     R("R5", [("chunk_size_abnormal", "==", True), ("answer_truncated", "==", True)], ("diagnosis", "CHUNKING_FAULT"), 8, 10, "abnormal chunk size + truncated answer"),
     R("R6", [("key_fact_split", "==", True)], ("diagnosis", "CHUNKING_FAULT"), 6, 11, "key fact split across chunks"),
     R("R7", [("outdated_evidence", "==", True)], ("diagnosis", "STALE_DOCUMENT"), 8, 12, "answer taken from superseded document"),
     R("R8", [("newer_version_exists", "==", True), ("doc_age_days", ">", 365)], ("diagnosis", "STALE_DOCUMENT"), 8, 13, "old document with newer version"),
     R("R9", [("answer_matches_old_value", "==", True)], ("diagnosis", "STALE_DOCUMENT"), 5, 14, "answer equals old value"),
     R("R10", [("retrieval_gap", "==", True), ("vocab_mismatch", "==", True)], ("diagnosis", "RETRIEVAL_MISS"), 8, 15, "vocabulary mismatch and wrong documents"),
     R("R11", [("retrieval_gap", "==", True), ("top_sim", "<", 0.2)], ("diagnosis", "RETRIEVAL_MISS"), 7, 16, "low similarity and wrong documents"),
     R("R12", [("top_sim", "<", 0.15)], ("diagnosis", "RETRIEVAL_MISS"), 6, 17, "very low similarity"),
     R("R13", [("answer_in_context", "==", False), ("top_sim", ">=", 0.4)], ("diagnosis", "HALLUCINATION"), 7, 18, "good retrieval but answer unsupported"),
     R("R14", [("answer_in_context", "==", False), ("citation_present", "==", False)], ("diagnosis", "HALLUCINATION"), 7, 19, "unsupported and uncited"),
     R("R15", [("answer_in_context", "==", False)], ("diagnosis", "HALLUCINATION"), 3, 20, "answer not in retrieved text"),
     R("R16", [("gold_doc_in_topk", "==", True), ("answer_in_context", "==", False)], ("diagnosis", "HALLUCINATION"), 7, 21, "right document retrieved, answer unsupported"),
    ]

BASE = dict(scanned_source=False, garbled_text=False, digits_wrong=False, ocr_conf=0.9, key_fact_split=False, answer_truncated=False, chunk_len=120,
            newer_version_exists=False, answer_matches_old_value=False, doc_age_days=100, top_sim=0.5, gold_doc_in_topk=True, vocab_mismatch=False,
            answer_in_context=True, citation_present=True)
def case(cid, label, drop=(), **kw):
    f = dict(BASE); f.update(kw)
    for d in drop: f.pop(d, None)
    return dict(id=cid, label=label, facts=f)
O, C, S, RM, H = "OCR_FAILURE", "CHUNKING_FAULT", "STALE_DOCUMENT", "RETRIEVAL_MISS", "HALLUCINATION"
# 30 labelled cases (synthetic, authored as a stand-in for expert labels): first 15 = TRAIN (used to write the rules), last 15 = HOLD-OUT
TRAIN = [
 case("t1", O, scanned_source=True, garbled_text=True, ocr_conf=0.45, digits_wrong=True), case("t2", O, scanned_source=True, garbled_text=True, ocr_conf=0.55, top_sim=0.47),
 case("t3", O, scanned_source=True, digits_wrong=True, ocr_conf=0.35, top_sim=0.6),
 case("t4", C, key_fact_split=True, answer_truncated=True, chunk_len=35), case("t5", C, answer_truncated=True, chunk_len=25, top_sim=0.45),
 case("t6", C, key_fact_split=True, chunk_len=520, top_sim=0.38),
 case("t7", S, newer_version_exists=True, answer_matches_old_value=True, doc_age_days=400, top_sim=0.6), case("t8", S, newer_version_exists=True, answer_matches_old_value=True, doc_age_days=800),
 case("t9", S, newer_version_exists=True, answer_matches_old_value=True, doc_age_days=500),
 case("t10", RM, top_sim=0.08, gold_doc_in_topk=False, vocab_mismatch=True, answer_in_context=False), case("t11", RM, top_sim=0.14, gold_doc_in_topk=False, answer_in_context=False),
 case("t12", RM, top_sim=0.18, gold_doc_in_topk=False, vocab_mismatch=True, answer_in_context=False),
 case("t13", H, top_sim=0.62, answer_in_context=False, citation_present=False), case("t14", H, top_sim=0.55, answer_in_context=False), case("t15", H, top_sim=0.45, answer_in_context=False, citation_present=False)]
HOLD = [
 case("h1", O, scanned_source=True, garbled_text=True, ocr_conf=0.5, digits_wrong=True), case("h2", O, scanned_source=True, garbled_text=True, drop=["ocr_conf"], top_sim=0.5),
 case("h3", O, scanned_source=True, ocr_conf=0.5, top_sim=0.3, answer_in_context=False),
 case("h4", C, key_fact_split=True, answer_truncated=True, chunk_len=30), case("h5", C, chunk_len=600, top_sim=0.42),
 case("h6", C, key_fact_split=True, top_sim=0.3, answer_in_context=False),
 case("h7", S, newer_version_exists=True, answer_matches_old_value=True, doc_age_days=250), case("h8", S, newer_version_exists=True, drop=["answer_matches_old_value"], doc_age_days=900),
 case("h9", S, answer_matches_old_value=True, drop=["newer_version_exists"], doc_age_days=300),
 case("h10", RM, top_sim=0.1, gold_doc_in_topk=False, vocab_mismatch=True, answer_in_context=False), case("h11", RM, top_sim=0.35, gold_doc_in_topk=False, vocab_mismatch=True, answer_in_context=False),
 case("h12", RM, top_sim=0.22, gold_doc_in_topk=False, answer_in_context=False, citation_present=False),
 case("h13", H, top_sim=0.58, answer_in_context=False, citation_present=False), case("h14", H, top_sim=0.35, answer_in_context=False), case("h15", H, top_sim=0.5, citation_present=False, drop=["answer_in_context"])]

# ---- adversarial cases + PREDICTIONS written BEFORE running (B2.2) -----------------------------------
ADV = [
 dict(id="ADV1 conflict OCR vs STALE", facts=dict(BASE, scanned_source=True, garbled_text=True, ocr_conf=0.5, newer_version_exists=True, answer_matches_old_value=True, doc_age_days=500),
      pred=dict(specificity="OCR_FAILURE", priority="OCR_FAILURE", recency="STALE_DOCUMENT")),
 dict(id="ADV2 missing evidence (only top_sim=0.1)", facts=dict(top_sim=0.1), pred=dict(specificity="RETRIEVAL_MISS", priority="RETRIEVAL_MISS", recency="RETRIEVAL_MISS")),
 dict(id="ADV3 order dependence (R11 vs R14 tie)", facts=dict(BASE, top_sim=0.1, gold_doc_in_topk=False, answer_in_context=False, citation_present=False),
      pred=dict(specificity="RETRIEVAL_MISS", priority="RETRIEVAL_MISS", recency="HALLUCINATION")),
 dict(id="ADV4 empty evidence", facts={}, pred=dict(specificity="UNKNOWN", priority="UNKNOWN", recency="UNKNOWN")),
 dict(id="ADV5 contradictory (garbled but not scanned)", facts=dict(garbled_text=True, scanned_source=False), pred=dict(specificity="UNKNOWN", priority="UNKNOWN", recency="UNKNOWN")),
 dict(id="ADV6a boundary top_sim=0.199", facts=dict(BASE, top_sim=0.199, gold_doc_in_topk=False, answer_in_context=False),
      pred=dict(specificity="RETRIEVAL_MISS", priority="RETRIEVAL_MISS", recency="HALLUCINATION")),
 dict(id="ADV6b boundary top_sim=0.201", facts=dict(BASE, top_sim=0.201, gold_doc_in_topk=False, answer_in_context=False),
      pred=dict(specificity="HALLUCINATION", priority="HALLUCINATION", recency="HALLUCINATION")),
]
FAULTS = [O, C, S, RM, H]
def evaluate(shell, cases, strategy="specificity"):
    rows = []
    for c in cases:
        r = shell.run(c["facts"], strategy); rows.append((c["id"], c["label"], r["diagnosis"], len(r["fired"]), r))
    return rows
def metrics(rows):
    acc = sum(l == p for _, l, p, _, _ in rows) / len(rows); per = {}
    for f in FAULTS:
        tp = sum(l == f and p == f for _, l, p, _, _ in rows); fp = sum(l != f and p == f for _, l, p, _, _ in rows); fn = sum(l == f and p != f for _, l, p, _, _ in rows)
        per[f] = dict(precision=tp / (tp + fp) if tp + fp else None, recall=tp / (tp + fn) if tp + fn else None, tp=tp, fp=fp, fn=fn)
    return acc, per

if __name__ == "__main__":
    sh = Shell(build_rules()); out = {}
    for name, cs in [("train", TRAIN), ("holdout", HOLD)]:
        for strat in ["specificity", "priority", "recency"]:
            rows = evaluate(sh, cs, strat); acc, per = metrics(rows)
            out[f"{name}/{strat}"] = dict(acc=acc, per=per, rules_fired=sum(r[3] for r in rows) / len(rows), diag={r[0]: r[2] for r in rows})
            print(name, strat, "acc %.2f" % acc, "rules/case %.2f" % out[f"{name}/{strat}"]["rules_fired"])
    for r in evaluate(sh, HOLD): print(r[0], r[1], "->", r[2], "OK" if r[1] == r[2] else "WRONG")
    print(sh.how(sh.run(HOLD[0]["facts"]))); print(sh.how(sh.run(HOLD[8]["facts"]))); print(sh.how(sh.run(HOLD[2]["facts"]))); print(sh.why("ocr_conf")); print(sh.why("gold_doc_in_topk"))
    # adversarial
    adv = []
    for a in ADV:
        row = dict(id=a["id"])
        for st in ["specificity", "priority", "recency"]:
            res = sh.run(a["facts"], st); row[st] = (a["pred"][st], res["diagnosis"], res["conflict"])
        adv.append(row); print(row)
    # order dependence: swap R11 and R14 list positions
    sh2 = Shell(build_rules()); i, j = [k for k, r in enumerate(sh2.rules) if r.id == "R11"][0], [k for k, r in enumerate(sh2.rules) if r.id == "R14"][0]
    sh2.rules[i], sh2.rules[j] = sh2.rules[j], sh2.rules[i]
    adv3 = ADV[2]["facts"]; order = {st: (sh.run(adv3, st)["diagnosis"], sh2.run(adv3, st)["diagnosis"]) for st in ["specificity", "priority", "recency", "order"]}
    print("ADV3 original-vs-swapped", order)
    # B2.3 knowledge acquisition at run time + regression
    allc = TRAIN + HOLD + [dict(id=a["id"].split(" ")[0], label="?", facts=a["facts"]) for a in ADV]
    before = {c["id"]: sh.run(c["facts"])["diagnosis"] for c in allc}
    sh.add_rule(dict(id="R17", **{"if": [("scanned_source", "==", True), ("answer_in_context", "==", False)]}, then=("diagnosis", "OCR_FAILURE"), prio=8, text="expert: scanned page whose words vanish from the answer context = OCR dropped text"))
    sh.add_rule(dict(id="R18", **{"if": [("chunk_size_abnormal", "==", True)]}, then=("diagnosis", "CHUNKING_FAULT"), prio=5, text="expert: abnormal chunk size alone indicates chunking fault"))
    sh.edit_rule("R15", **{"if": [("answer_in_context", "==", False), ("top_sim", ">=", 0.25)]})
    after = {c["id"]: sh.run(c["facts"])["diagnosis"] for c in allc}
    changed = {k: (before[k], after[k]) for k in before if before[k] != after[k]}; print("REGRESSION changed:", changed)
    labels = {c["id"]: c["label"] for c in allc}
    hold_after = sum(after[c["id"]] == c["label"] for c in HOLD) / 15; print("holdout acc after edit", hold_after)
    json.dump(dict(out=out, adv=adv, order=order, changed=changed, hold_after=hold_after, labels=labels), open("out/expert_results.json", "w"), indent=1, default=str)
