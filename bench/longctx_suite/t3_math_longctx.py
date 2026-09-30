#!/usr/bin/env python3
"""T3: hard math (MATH-500 level 5) behind a fixed ~40k-token code context, thinking on, server default
sampling, fixed seeds. Scores correct / wrong / DNF (hit max_tokens or no \\boxed answer) / loop (a 200-char
span of the reasoning repeated >=4 times). usage: t3_math_longctx.py <tag> [--n 40] [--conc 2]"""
import json, sys, os, re, glob, random, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(HERE, "..", "quality-data")
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
tag = sys.argv[1]
N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 40
CONC = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 2
MAXTOK = 16000

probs = [json.loads(l) for l in open(f"{Q}/math500/test.jsonl")]
probs = [p for p in probs if p["level"] == 5]
random.Random(7).shuffle(probs); probs = probs[:N]
src = sorted(glob.glob(os.path.join(HERE, "..", "..", "venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
CTX = "".join(open(p, errors="ignore").read() for p in src)[900000:900000 + 150000]   # ~40k tokens, identical for all

def norm(s):
    s = (s or "").strip().replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("^{\\circ}", "").replace("\\$", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s).rstrip(".")
    return s

def boxed(text):
    i = (text or "").rfind("\\boxed{")
    if i < 0: return None
    j = i + 7; depth = 1
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0); j += 1
    return text[i + 7:j - 1] if depth == 0 else None

def looped(text, span=200, reps=4):
    t = text or ""
    if len(t) < span * reps: return False
    step = span // 2
    seen = {}
    for i in range(0, len(t) - span, step):
        k = t[i:i + span]; seen[k] = seen.get(k, 0) + 1
        if seen[k] >= reps: return True
    return False

def run(i):
    p = probs[i]
    msgs = [{"role": "system", "content": "You are a coding and math assistant. The user's repository context is below; it is unrelated to the question.\n\n<repository>\n" + CTX + "\n</repository>"},
            {"role": "user", "content": p["problem"] + "\n\nPut your final answer in \\boxed{}."}]
    body = {"model": "qwen3.8-27b", "messages": msgs, "max_tokens": MAXTOK, "seed": 5000 + i}
    t0 = time.time()
    try:
        r = json.load(urllib.request.urlopen(urllib.request.Request(API + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}), timeout=7200))
    except Exception as e:
        return {"i": i, "status": "ERROR", "err": str(e)[:200]}
    ch = r["choices"][0]; m = ch["message"]
    reasoning = m.get("reasoning_content") or m.get("reasoning") or ""
    ans = boxed(m.get("content")) or boxed(reasoning[-3000:])
    lp = looped(reasoning)
    if ch.get("finish_reason") == "length" or ans is None:
        st = "DNF"
    else:
        st = "CORRECT" if norm(ans) == norm(p["answer"]) else "WRONG"
    return {"i": i, "status": st, "loop": lp, "finish": ch.get("finish_reason"), "answer": ans, "gold": p["answer"],
            "completion_tokens": r["usage"]["completion_tokens"], "secs": round(time.time() - t0, 1)}

res = []
with ThreadPoolExecutor(CONC) as ex:
    for out in ex.map(run, range(len(probs))):
        res.append(out); print(tag, out["i"], out["status"], "loop" if out.get("loop") else "", out.get("completion_tokens"), out.get("answer"), "| gold", out.get("gold"), flush=True)
summ = {s: sum(1 for r in res if r["status"] == s) for s in ("CORRECT", "WRONG", "DNF", "ERROR")}
summ["loops"] = sum(1 for r in res if r.get("loop"))
summ["mean_completion_tokens"] = round(sum(r.get("completion_tokens", 0) for r in res) / len(res))
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"tag": tag, "summary": summ, "items": res}, open(os.path.join(HERE, "results", f"t3-{tag}.json"), "w"), indent=1)
print("SUMMARY", tag, summ)
