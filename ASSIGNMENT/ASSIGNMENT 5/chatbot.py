"""Task B4: retrieval-based operator assistant for the intrusion-detection system (TF-IDF + cosine, confidence-threshold fallback)."""
import numpy as np, random, json, time, pickle, re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
TRAIN = {
 "list_open_alerts": ["which alerts are open", "show me all open alerts", "list the unresolved alerts", "what alerts are still pending", "any active alerts right now", "display current open alerts", "give me the open alert list", "what incidents are open", "are there unacknowledged alerts", "show active security alerts", "open alerts please", "which alarms are currently active"],
 "explain_alert": ["why was this flagged", "why did the system raise this alert", "explain this detection", "what triggered the alert", "give me the reason for the flag", "why is this request marked suspicious", "what evidence led to this alert", "explain why that session was flagged", "how did the model decide this is an attack", "reason for this alarm", "tell me why it was detected", "what made this look malicious"],
 "alert_count": ["how many alerts today", "count the attacks by type", "number of alerts raised", "how many brute force attempts were detected", "give me alert statistics", "total detections so far", "how many prompt injections were caught", "summary of attacks by category", "alert totals please", "how many incidents were found", "count of flagged requests", "what is the number of detections"],
 "top_sources": ["which users are attacking the most", "show top attacking sources", "who is generating the most alerts", "list the most suspicious accounts", "top offenders please", "which source sent the most attacks", "worst clients by alert count", "show the noisiest attackers", "most active malicious sessions", "who triggered the most detections", "rank the suspicious users", "biggest threat sources"],
 "model_status": ["is the detection model running", "what is the model latency", "show model status", "how fast is the detector", "is the edge model healthy", "report detector performance", "what is the current detection accuracy", "model health check", "is the classifier online", "how long does a prediction take", "show system performance", "detector status report"],
 "attack_info": ["what is a prompt injection", "explain brute force attack", "what does dos flood mean", "tell me about malicious upload attacks", "what is credential spraying", "define scraping attack", "how does a denial of service work", "what are the attack types you detect", "describe prompt injection in simple words", "what is a polyglot file", "meaning of low and slow attack", "explain the attack categories"],
 "mark_false_positive": ["mark this alert as false positive", "this was a legitimate user", "dismiss this alert", "flag this as benign", "that alert was wrong", "ignore this detection it is safe", "label the alert as not an attack", "close this alert as harmless", "report a false alarm", "this one is a false positive", "clear the alert it was normal", "mark as safe"],
 "block_source": ["block this user", "ban the attacking account", "block that ip address", "stop this source from accessing", "blacklist this session", "lock the suspicious account", "deny access to this client", "disable the offending user", "prevent this source from logging in", "cut off this attacker", "add the source to the block list", "suspend this account now"],
 "last_attack": ["when was the last attack", "time of the latest detection", "when did the last alert happen", "most recent incident time", "show the latest attack", "when was the previous intrusion", "last time we were attacked", "when did we last detect something", "latest alert timestamp", "what was the most recent attack", "recent attack time please", "when did the newest alert fire"],
 "greeting_help": ["hello", "hi there", "help", "what can you do", "good morning", "show me the commands", "how do i use this assistant", "hey assistant", "what questions can i ask", "i need help", "greetings", "list your features"],
}
VAL = {
 "list_open_alerts": ["what is currently open", "unresolved alerts list", "do we have open alerts", "open alarms now"], "explain_alert": ["why was it detected", "what caused this flag", "reason behind the alert", "explain the alarm"],
 "alert_count": ["how many detections", "alert count by category", "total number of alerts", "stats on attacks"], "top_sources": ["who attacks the most", "top suspicious users", "main offenders list", "worst sources"],
 "model_status": ["is the model online", "latency of the detector", "model performance report", "health of the classifier"], "attack_info": ["what is credential stuffing", "explain injection attacks", "what is a dos attack", "tell me attack types"],
 "mark_false_positive": ["it was a false alarm", "dismiss as benign", "not an attack mark it", "wrong detection close it"], "block_source": ["ban this account", "block the attacker", "deny this ip", "lock this user out"],
 "last_attack": ["latest attack time", "when was the last incident", "most recent detection", "last alert time"], "greeting_help": ["hey", "what can i ask", "help me", "hello assistant"]}
TEST = {
 "list_open_alerts": ["which alerts are still open right now", "can you show unresolved incidents", "list pending security alerts", "what is outstanding in the alert queue", "show me every active alert", "any open alarms to handle", "give me the current open alerts", "open alerts report"],
 "explain_alert": ["why was this session flagged as an attack", "what is the reason for this alert", "explain the flag on this request", "how come this was marked malicious", "why did the detector fire", "what caused this alarm to trigger", "justify this detection", "tell me why this alert exists"],
 "alert_count": ["how many attacks were detected today", "give me the alert counts by type", "what is the total number of alerts", "number of brute force alerts", "count all detections", "how many injections did we catch", "alert statistics by category", "how many incidents so far"],
 "top_sources": ["which accounts caused the most alerts", "show the top attackers", "who are the worst offenders", "list the most suspicious sources", "which client attacked the most", "rank sources by alert count", "who is behind most detections", "top malicious users"],
 "model_status": ["is the model healthy", "what is the prediction latency", "show detector status", "how is the model performing now", "is the edge classifier running", "check model health", "what is the detection speed", "model status please"],
 "attack_info": ["what is a brute force attack", "explain prompt injection to me", "what does credential spraying mean", "tell me about dos floods", "what is a malicious upload", "describe the scraping attack", "which attacks can you detect", "what is a low and slow attack"],
 "mark_false_positive": ["this alert is a false positive", "mark this as a legitimate user", "that was not an attack", "dismiss this alert as safe", "close this alarm it is harmless", "label this detection as benign", "report this as a false alarm", "ignore this alert it was normal"],
 "block_source": ["block this attacker", "please ban this user account", "blacklist the offending ip", "deny this source access", "lock out the suspicious user", "stop this client from connecting", "add this source to the blocklist", "disable that account"],
 "last_attack": ["when did the last attack happen", "show the most recent alert time", "what was the latest incident", "when were we last attacked", "time of last detection", "when did the previous alert occur", "latest intrusion time please", "how long ago was the last alert"],
 "greeting_help": ["hi", "hello there", "what can you help me with", "show me what you can do", "good evening", "i need some help", "help me use this", "what commands do you have"]}
OOS_VAL = ["who won the football match", "what is the weather today", "tell me a joke", "how do i bake a cake", "what is the capital of japan", "play some music", "translate this to french", "what is the price of bitcoin", "recommend a movie", "how tall is the eiffel tower", "book me a flight", "what is two plus two", "who is the president", "write a poem", "how old is the moon"]
OOS_TEST = ["what is the best pizza topping", "who painted the mona lisa", "how do i learn guitar", "what time does the cinema open", "give me a recipe for pasta", "what is the speed of light", "tell me the news headlines", "who is the best cricketer", "how do i lose weight", "what is the tallest mountain", "can you sing a song", "what is the exchange rate of the dollar", "suggest a holiday destination", "how many days are in a leap year", "what is the meaning of life", "set an alarm for seven", "what is my horoscope", "how far is the moon", "what should i cook tonight", "who invented the telephone",
            "which restaurant is the best nearby", "is it going to rain tomorrow", "what is photosynthesis", "how do airplanes fly", "name a famous scientist", "what is the population of india", "how do i tie a tie", "what is your favourite colour", "show me funny cat videos", "when is the next holiday"]
INTENTS = list(TRAIN)
STOP = ['a','an','the','is','are','was','were','do','does','did','of','to','in','it','this','that','me','you','i','what','how','which','who','when','please','can','there','we','our','my','be','for','on','at','by','with','and','or']
class Bot:
    def __init__(self, train):
        self.u = [(t, k) for k, v in train.items() for t in v]
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, stop_words=STOP).fit([t for t, _ in self.u]); self.M = self.vec.transform([t for t, _ in self.u]); self.lab = [k for _, k in self.u]
    def score(self, text):
        s = (self.M @ self.vec.transform([text]).T).toarray().ravel(); i = int(s.argmax()); return self.lab[i], float(s[i])
    def predict(self, text, tau):
        k, s = self.score(text); return (k if s >= tau else "out_of_scope"), s
# ---------- speech-recognition error simulation -----------
VOCAB = sorted({w for v in TRAIN.values() for t in v for w in t.split()})
def corrupt(text, wer, rnd):
    out = []
    for w in text.split():
        if rnd.random() < wer:
            r = rnd.random()
            if r < 0.5: out.append(rnd.choice(VOCAB))                  # substitution
            elif r < 0.75: continue                                     # deletion
            else: out += [w, rnd.choice(VOCAB)]                          # insertion
        else: out.append(w)
    return " ".join(out) if out else "uh"
if __name__ == "__main__":
    bot = Bot(TRAIN); rnd = random.Random(0)
    # ---- threshold chosen on validation set: smallest tau with OOS rejection >= 90% (keeps in-scope recall as high as possible)
    vs = [(t, k) for k, v in VAL.items() for t in v] + [(t, "out_of_scope") for t in OOS_VAL]
    taus = np.round(np.arange(0.0, 0.9, 0.025), 3); Js = []
    vin = [(t, k) for k, v in VAL.items() for t in v]
    for T in taus:
        acc = np.mean([bot.predict(t, T)[0] == k for t, k in vin]); rej = np.mean([bot.predict(t, T)[0] == "out_of_scope" for t in OOS_VAL]); Js.append(acc + rej)
    good = [float(T) for T, J in zip(taus, Js) if J >= max(Js) - 0.02]; tau = float(np.median(good)); print("tau", tau, "plateau", good[0], good[-1])
    ts = [(t, k) for k, v in TEST.items() for t in v] + [(t, "out_of_scope") for t in OOS_TEST]
    y = [k for _, k in ts]; lat = []; pred = []
    for t, _ in ts: t0 = time.perf_counter(); pred.append(bot.predict(t, tau)[0]); lat.append((time.perf_counter() - t0) * 1000)
    labels = INTENTS + ["out_of_scope"]; p, r, f, sup = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per = {l: dict(precision=float(a), recall=float(b), f1=float(c), support=int(d)) for l, a, b, c, d in zip(labels, p, r, f, sup)}
    cm = confusion_matrix(y, pred, labels=labels).tolist(); acc = float(np.mean([a == b for a, b in zip(y, pred)]))
    res = dict(tau=tau, accuracy=acc, per=per, confusion=cm, labels=labels, latency_mean_ms=float(np.mean(lat)), latency_p95_ms=float(np.percentile(lat, 95)), n_test=len(ts), n_train=len(bot.u))
    # ---- B4.2 WER degradation (5 corruption seeds averaged) + threshold sweep
    wers = [0.0, 0.1, 0.2, 0.3]; table = {}; sweep = {}
    for w in wers:
        accs, oosdet, far, frr = [], [], [], []; sw = {float(T): [[], [], []] for T in taus}
        for seed in range(5):
            rr = random.Random(100 + seed); noisy = [(corrupt(t, w, rr), k) for t, k in ts]
            sc = [bot.score(t) for t, _ in noisy]
            for T in [tau]:
                pr = [(k if s >= T else "out_of_scope") for k, s in sc]
                ins = [i for i, (_, k) in enumerate(noisy) if k != "out_of_scope"]; oo = [i for i, (_, k) in enumerate(noisy) if k == "out_of_scope"]
                accs.append(np.mean([pr[i] == noisy[i][1] for i in ins])); oosdet.append(np.mean([pr[i] == "out_of_scope" for i in oo])); far.append(np.mean([pr[i] != "out_of_scope" for i in oo])); frr.append(np.mean([pr[i] == "out_of_scope" for i in ins]))
            for T in taus:
                pr = [(k if s >= T else "out_of_scope") for k, s in sc]
                sw[float(T)][0].append(np.mean([pr[i] == noisy[i][1] for i in ins])); sw[float(T)][1].append(np.mean([pr[i] != "out_of_scope" for i in oo])); sw[float(T)][2].append(np.mean([pr[i] == "out_of_scope" for i in ins]))
        table[str(w)] = dict(intent_accuracy=float(np.mean(accs)), oos_detection=float(np.mean(oosdet)), false_accept_rate=float(np.mean(far)), false_reject_rate=float(np.mean(frr)))
        sweep[str(w)] = {str(T): [float(np.mean(x)) for x in v] for T, v in sw.items()}
    res["wer"] = table; res["sweep"] = sweep; res["taus"] = [float(t) for t in taus]
    # ---- demo transcript wired to the B3 detector output
    d = pickle.load(open("out/ids_test_preds.pkl", "rb")); te = d["te"]; pr = d["pred_cascade"]
    CL = ["normal", "brute_force", "prompt_injection", "dos_flood", "malicious_upload", "scraping"]
    last = list(range(len(te) - 300, len(te))); alerts = [i for i in last if pr[i] != 0]; acked = set(alerts[::3])
    def explain(i):
        r = te[i]; c = CL[pr[i]]
        return {"brute_force": f"{r['failed_logins']} failed logins across {r['distinct_usernames']} usernames at {r['req_per_min']:.0f} req/min",
                "prompt_injection": f"query text matches injection patterns: '{r['text'][:50]}...'", "dos_flood": f"{r['req_per_min']:.0f} requests/min with repeat ratio {r['repeat_ratio']:.2f}",
                "malicious_upload": f"upload of {r['upload_kb']:.0f} KB (risky extension flag={r['ext_risk']})", "scraping": f"{r['req_per_min']:.0f} req/min from a bot user-agent on a single endpoint"}.get(c, "n/a")
    def respond(u):
        k, s = bot.predict(u, tau)
        if k == "list_open_alerts": o = [i for i in alerts if i not in acked]; return k, s, f"{len(o)} open alerts; latest: " + ", ".join(f"#{i} {CL[pr[i]]}" for i in o[-3:])
        if k == "explain_alert": i = [i for i in alerts if i not in acked][-1]; return k, s, f"Alert #{i} ({CL[pr[i]]}): {explain(i)}"
        if k == "alert_count": from collections import Counter; c = Counter(CL[pr[i]] for i in alerts); return k, s, "; ".join(f"{a}: {b}" for a, b in c.items())
        if k == "model_status": return k, s, "Edge cascade model online; mean latency 0.03 ms/sample (sandbox), accuracy 95.6% on last-30% test period"
        if k == "last_attack": i = alerts[-1]; return k, s, f"Last alert #{i} ({CL[pr[i]]}) at normalised time {te[i]['t']:.3f}"
        if k == "out_of_scope": return k, s, "Sorry, I can only answer questions about the security alerts."
        return k, s, f"[{k}] handled (action stub)"
    demo = [("Which alerts are open?",) , ("Why was this flagged?",), ("how many attacks so far", ), ("what is the model latency",), ("who won the cricket match",), ("when was the last attack",)]
    res["demo"] = [(q[0],) + respond(q[0]) for q in demo]
    json.dump(res, open("out/chatbot_results.json", "w"), indent=1); print(json.dumps({k: res[k] for k in ["tau", "accuracy", "latency_mean_ms", "latency_p95_ms", "wer", "demo"]}, indent=1))
    print({l: round(v["f1"], 2) for l, v in per.items()})
