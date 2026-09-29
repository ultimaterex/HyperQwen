"""HTTP-server variant of gen_data.py: uses this repo's own served checkpoint
(via llama-swap / the OpenAI-compatible chat completions endpoint) instead of an
in-process vllm.LLM(), since this box only has the Docker serving setup, not a
native venv with this repo's vLLM patches applied.

Reuses drafter/data/prompts.jsonl (built by collect_prompts.py). Requires the
target model already booted and reachable, and that vLLM's OpenAI server
supports `return_token_ids` (confirmed on vLLM 0.28.0 here) so we get real
output token ids back, not just decoded text.

Usage:
  quant-venv/bin/python drafter/gen_data_http.py --model qwen-3.8-27b-syv-swift-unc \
    --url http://localhost:8080/v1/chat/completions [--limit N] [--concurrency 4]
"""
import argparse
import json
import os
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(HERE, "data")

parser = argparse.ArgumentParser()
parser.add_argument("--model", required=True)
parser.add_argument("--url", default="http://localhost:8080/v1/chat/completions")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--concurrency", type=int, default=1)
parser.add_argument("--max-think", type=int, default=2048)
parser.add_argument("--max-nothink", type=int, default=1024)
args = parser.parse_args()

prompts = [json.loads(l) for l in open(f"{D}/prompts.jsonl")]
if args.limit:
    prompts = prompts[: args.limit]

out_path = f"{D}/gen.jsonl"
done = set()
if os.path.exists(out_path):
    for l in open(out_path):
        try:
            done.add(json.loads(l)["id"])
        except Exception:
            pass
todo = [p for p in prompts if p["id"] not in done]
print(f"{len(prompts)} prompts, {len(done)} done, {len(todo)} to go", flush=True)

lock = threading.Lock()
out_f = open(out_path, "a")
ntok = 0
t0 = time.time()
n_done = 0


def gen_one(p):
    global ntok, n_done
    max_tokens = args.max_think if p["think"] else args.max_nothink
    body = json.dumps({
        "model": args.model,
        "messages": p["messages"],
        "max_tokens": max_tokens,
        "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": p["think"]},
    }).encode()
    req = urllib.request.Request(args.url, data=body, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            d = json.loads(resp.read())
        choice = d["choices"][0]
        rec = {
            "id": p["id"], "src": p["src"], "think": p["think"],
            "output_ids": choice["token_ids"],
            "finish": choice["finish_reason"],
        }
        with lock:
            out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            ntok += len(rec["output_ids"])
            n_done += 1
            if n_done % 25 == 0:
                el = time.time() - t0
                print(f"[{n_done}/{len(todo)}] {ntok} output tokens, {ntok/el:.0f} tok/s, {el/60:.1f} min", flush=True)
    except Exception as e:
        print("error", p["id"], e, flush=True)


if args.concurrency <= 1:
    for p in todo:
        gen_one(p)
else:
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        list(ex.map(gen_one, todo))

out_f.close()
print("done", n_done, "generated,", ntok, "tokens")
