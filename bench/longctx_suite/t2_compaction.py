#!/usr/bin/env python3
"""T2: state tracking across compaction cycles. Each episode: 10 integer registers, CYCLES cycles; every
cycle hides OPS operations inside ~CHUNK_TOK tokens of noisy tool-output logs. The model only ever sees its
own previous state summary + the new chunk (compaction), so errors compound. Thinking on, server default
sampling, fixed seeds. usage: t2_compaction.py <tag> [--episodes N] [--conc 2]"""
import json, sys, os, re, random, time, threading, urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
tag = sys.argv[1]
EPISODES = int(sys.argv[sys.argv.index("--episodes") + 1]) if "--episodes" in sys.argv else 12
CONC = int(sys.argv[sys.argv.index("--conc") + 1]) if "--conc" in sys.argv else 2
CYCLES, OPS, CHUNK_TOK, MAXTOK = 6, 8, 16000, 12000
if "--hard" in sys.argv:
    CYCLES, OPS, CHUNK_TOK, MAXTOK = 10, 16, 24000, 16000
REGS = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet"]

WORDS = ("request handler cache miss retry worker queue shard index flush commit rollback socket timeout "
         "latency batch token stream parser schema migration replica leader follower heartbeat").split()
def noise_line(rng):
    kind = rng.random()
    ts = f"2026-09-{rng.randint(1,28):02d}T{rng.randint(0,23):02d}:{rng.randint(0,59):02d}:{rng.randint(0,59):02d}Z"
    if kind < 0.5:
        return f"{ts} INFO {rng.choice(WORDS)}.{rng.choice(WORDS)} id={rng.randint(1000,99999)} dur={rng.randint(1,900)}ms " + " ".join(rng.choice(WORDS) for _ in range(rng.randint(4, 12)))
    if kind < 0.75:
        return f"{ts} WARN {rng.choice(WORDS)} value={rng.randint(-500,500)} limit={rng.randint(0,999)} ({rng.choice(REGS)}_cache stale)"
    return f"    at {rng.choice(WORDS)}.{rng.choice(WORDS)}({rng.choice(WORDS)}.py:{rng.randint(1,2000)})"

def make_op(rng, st):
    r = rng.choice(REGS); t = rng.random()
    if t < 0.35:
        k = rng.randint(1, 40); st[r] += k; return f"{r} += {k}"
    if t < 0.6:
        k = rng.randint(1, 40); st[r] -= k; return f"{r} -= {k}"
    if t < 0.75:
        o = rng.choice([x for x in REGS if x != r]); st[r], st[o] = st[o], st[r]; return f"swap {r} {o}"
    if t < 0.9:
        o = rng.choice([x for x in REGS if x != r]); st[r] = st[o] + st[r]; return f"{r} = {r} + {o}"
    o = rng.choice([x for x in REGS if x != r]); k = rng.randint(1, 20)
    if st[o] > 50: st[r] -= k
    return f"if {o} > 50 then {r} -= {k}"

def episode_plan(seed):
    rng = random.Random(seed)
    st = {r: rng.randint(0, 100) for r in REGS}
    init = dict(st); cycles = []
    for c in range(CYCLES):
        lines = [noise_line(rng) for _ in range(int(CHUNK_TOK / 28))]
        ops = []
        for pos in sorted(rng.sample(range(len(lines)), OPS)):
            op = make_op(rng, st); ops.append(op)
            lines[pos] = lines[pos] + f"\n[STATE-OP] {op}"
        cycles.append({"chunk": "\n".join(lines), "ops": ops, "expected": dict(st)})
    return init, cycles

SYS = ("You maintain a set of integer registers. You will receive the current register values and a chunk of "
       "tool output logs. Some log lines are followed by a line starting with [STATE-OP]: apply exactly those "
       "operations, in order, to the registers (ignore everything else, including values that merely appear in logs). "
       "Operations: 'x += k', 'x -= k', 'swap x y', 'x = x + y' (uses current values), "
       "'if y > 50 then x -= k' (check y at that moment). Reply with the final register values as a single JSON object "
       "on the last line, e.g. {\"alpha\": 1, ...}. Nothing after the JSON.")

def chat(messages, seed):
    body = {"model": "qwen3.8-27b", "messages": messages, "max_tokens": MAXTOK, "seed": seed}
    r = urllib.request.Request(API + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=3600))

def parse_state(text):
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text or ""))):
        try:
            d = json.loads(m.group(0))
            if all(k in d for k in REGS): return {k: int(d[k]) for k in REGS}
        except Exception:
            pass
    return None

def run_episode(ep):
    seed = 1000 + ep
    init, cycles = episode_plan(seed)
    state = init; log = []
    for c, cy in enumerate(cycles):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": f"Current registers:\n{json.dumps(state)}\n\nTool output chunk {c+1}/{CYCLES}:\n{cy['chunk']}"}]
        t0 = time.time()
        try:
            r = chat(msgs, seed * 100 + c)
        except Exception as e:
            log.append({"cycle": c, "error": str(e)[:200]}); return {"ep": ep, "status": "ERROR", "cycles": log}
        ch = r["choices"][0]; msg = ch["message"]
        new = parse_state(msg.get("content"))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        entry = {"cycle": c, "finish": ch.get("finish_reason"), "completion_tokens": r["usage"]["completion_tokens"],
                 "reasoning_chars": len(reasoning), "secs": round(time.time() - t0, 1),
                 "correct": new == cy["expected"], "wrong_regs": None if new is None else sum(new[k] != cy["expected"][k] for k in REGS)}
        log.append(entry)
        if new is None:
            return {"ep": ep, "status": "DNF" if ch.get("finish_reason") == "length" else "NO_STATE", "cycles": log}
        state = new
    final_ok = state == cycles[-1]["expected"]
    return {"ep": ep, "status": "CORRECT" if final_ok else "WRONG", "cycles": log}

res = []
with ThreadPoolExecutor(CONC) as ex:
    for out in ex.map(run_episode, range(EPISODES)):
        res.append(out)
        print(tag, "ep", out["ep"], out["status"], [ (c.get("correct"), c.get("completion_tokens"), c.get("finish")) for c in out["cycles"]], flush=True)
summ = {s: sum(1 for r in res if r["status"] == s) for s in ("CORRECT", "WRONG", "DNF", "NO_STATE", "ERROR")}
cyc = [c for r in res for c in r["cycles"] if "correct" in c]
summ["cycle_accuracy"] = round(sum(c["correct"] for c in cyc) / len(cyc), 3) if cyc else None
summ["length_stops"] = sum(1 for c in cyc if c["finish"] == "length")
summ["mean_completion_tokens"] = round(sum(c["completion_tokens"] for c in cyc) / len(cyc)) if cyc else None
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump({"tag": tag, "summary": summ, "episodes": res}, open(os.path.join(HERE, "results", f"t2-{tag}.json"), "w"), indent=1)
print("SUMMARY", tag, summ)
