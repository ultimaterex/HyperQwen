"""Count token-id frequencies directly from drafter/data/gen.jsonl's output_ids
(no re-tokenization needed, we already have exact ids from generation) and
build a draft_vocab_ids.json in the same 90/10 held-out methodology as
prepare/build_draft_vocab.py's --corpus path.

Usage:
  quant-venv/bin/python drafter/count_vocab_from_gen.py <model-dir> [--n 40960] [--out FILE]
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

parser = argparse.ArgumentParser()
parser.add_argument("model_dir")
parser.add_argument("--n", type=int, default=40960)
parser.add_argument("--out", default=os.path.join(HERE, "draft_vocab_ids_swift_unc.json"))
parser.add_argument("--gen", default=os.path.join(HERE, "data", "gen.jsonl"))
args = parser.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(args.model_dir)

counts = collections.Counter()
held = collections.Counter()
total = 0
n_records = 0
with open(args.gen) as f:
    for j, line in enumerate(f):
        rec = json.loads(line)
        ids = rec["output_ids"]
        (held if j % 10 == 0 else counts).update(ids)
        total += len(ids)
        n_records += 1

print(f"{n_records} generation records, {total} output tokens")

special = set(tok.all_special_ids)
for name in ("<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
             "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"):
    tid = tok.convert_tokens_to_ids(name)
    if isinstance(tid, int) and tid >= 0:
        special.add(tid)

top = [t for t, _ in counts.most_common() if t not in special][: args.n - len(special)]
ids = sorted(set(top) | special)
cover = sum(c for t, c in held.items() if t in set(ids)) / max(1, sum(held.values()))
print(f"draft vocab: {len(ids)} ids, held-out token coverage {cover*100:.2f}%")
print(f"distinct tokens ever emitted: {len(counts) + len(held)}")
for n_try in (16384, 25879, 32768, 40960, 49152):
    s = set(t for t, _ in counts.most_common(n_try)) | special
    c = sum(c for t, c in held.items() if t in s) / max(1, sum(held.values()))
    print(f"  coverage at N={n_try}: {c*100:.2f}%")

json.dump(ids, open(args.out, "w"))
print(f"written to {args.out}")

# for comparison: coverage of the SHIPPED (official-model) draft vocab against our own outputs
shipped_path = os.path.join(os.path.dirname(HERE), "prepare", "draft_vocab_ids.json")
if os.path.exists(shipped_path):
    shipped = set(json.load(open(shipped_path)))
    cover_shipped = sum(c for t, c in held.items() if t in shipped) / max(1, sum(held.values()))
    print(f"shipped (official-model) vocab coverage on OUR outputs: {cover_shipped*100:.2f}%")
