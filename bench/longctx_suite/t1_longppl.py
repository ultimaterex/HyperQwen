#!/usr/bin/env python3
"""T1: teacher-forced NLL by context depth on long documents (needs a server with prompt_logprobs headroom,
SPEC=off, PREFIX_CACHE=0). usage: t1_longppl.py <tag> [--docs N]
Writes results/t1-<tag>.json: per-doc, per-bucket mean NLL over the same token positions."""
import json, sys, os, glob, math, urllib.request
import pyarrow.parquet as pq

HERE = os.path.dirname(os.path.abspath(__file__)); Q = os.path.join(HERE, "..", "quality-data")
API = os.environ.get("VLLM_API", "http://127.0.0.1:18020/v1")
tag = sys.argv[1]
NDOCS = int(sys.argv[sys.argv.index("--docs") + 1]) if "--docs" in sys.argv else 8
TARGET = 64000
BUCKETS = [(0, 2000), (2000, 8000), (8000, 16000), (16000, 32000), (32000, 48000), (48000, 64000)]

def post(path, body, timeout=3600):
    r = urllib.request.Request(API + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=timeout))

def docs():
    books = [t for t in pq.read_table(f"{Q}/pg19/test.parquet").column("text").to_pylist() if len(t) > 320000]
    out = [("pg19-%d" % i, b[:330000]) for i, b in enumerate(books[: NDOCS - 2])]
    src = sorted(glob.glob(os.path.join(HERE, "..", "..", "venv/lib/python3.12/site-packages/vllm/**/*.py"), recursive=True))
    code = "".join(open(p, errors="ignore").read() for p in src)
    out += [("code-a", code[200000:200000 + 260000]), ("code-b", code[2000000:2000000 + 260000])]
    return out

res = {"tag": tag, "docs": {}}
for name, text in docs():
    r = post("/completions", {"model": "qwen3.8-27b", "prompt": text, "max_tokens": 1, "temperature": 0, "prompt_logprobs": 0,
                              "truncate_prompt_tokens": TARGET})
    lp = r["choices"][0]["prompt_logprobs"]
    toks = r["choices"][0].get("prompt_token_ids") or []
    vals = []
    for i, d in enumerate(lp):
        if d is None: vals.append(None); continue
        # prompt_logprobs=0 returns just the actual token; take the entry whose rank/decoded matches
        v = next(iter(d.values()))
        vals.append(-v["logprob"] if isinstance(v, dict) else -v)
    b = {}
    for lo, hi in BUCKETS:
        seg = [x for x in vals[lo:hi] if x is not None and math.isfinite(x)]
        b[f"{lo//1000}-{hi//1000}k"] = round(sum(seg) / len(seg), 5) if seg else None
    res["docs"][name] = {"n_tokens": len(vals), "buckets": b}
    print(tag, name, len(vals), b, flush=True)

agg = {}
for k in res["docs"][next(iter(res["docs"]))]["buckets"]:
    xs = [d["buckets"][k] for d in res["docs"].values() if d["buckets"].get(k) is not None]
    agg[k] = round(sum(xs) / len(xs), 5) if xs else None
res["mean_nll_by_bucket"] = agg
os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
json.dump(res, open(os.path.join(HERE, "results", f"t1-{tag}.json"), "w"), indent=1)
print("MEAN", tag, agg)
