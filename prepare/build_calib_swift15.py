"""Calibration set for the Swift-1.5 uncensored AutoRound quant: 128 samples x 4096 tokens.

- 96 samples packed from drafter/data/gen.jsonl (this model family's own chat-templated generations, as token ids),
  English sources only (ultrachat / magicoder / gsm8k), thinking-trace samples first. Whole conversations are
  packed back to back and the last one is cut at the boundary.
- 32 samples packed from NeelNanda/pile-10k (the original recipe's generic text), tokenized with the model's
  tokenizer.
Output: a list of [1, 4096] int64 tensors saved with torch.save; AutoRound uses a non-str dataset as the dataloader
directly and skips anything shorter than seqlen.
usage: quant-venv/bin/python prepare/build_calib_swift15.py <model_dir> <out.pt>"""
import json, random, sys
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

model_dir, out = sys.argv[1], sys.argv[2]
SEQ, N_GEN, N_PILE = 4096, 96, 32
rng = random.Random(42)

rows = [json.loads(l) for l in open("drafter/data/gen.jsonl")]
rows = [r for r in rows if r["src"] in ("ultrachat", "magicoder", "gsm8k") and r.get("finish") in ("stop", "length", None)]
think = [r for r in rows if r["think"]]; plain = [r for r in rows if not r["think"]]
rng.shuffle(think); rng.shuffle(plain)
stream_rows = think + plain           # reasoning traces first; plain chat fills the rest

def pack(seqs, n):
    out, buf = [], []
    for s in seqs:
        buf.extend(s)
        while len(buf) >= SEQ and len(out) < n:
            out.append(torch.tensor([buf[:SEQ]], dtype=torch.long)); buf = buf[SEQ:]
        if len(out) >= n:
            break
    return out

gen = pack((r["prompt_ids"] + r["output_ids"] for r in stream_rows), N_GEN)
tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
pile = load_dataset("NeelNanda/pile-10k", split="train").shuffle(seed=42)
pile_ids = (tok(t["text"])["input_ids"] + [tok.eos_token_id] for t in pile)
pl = pack(pile_ids, N_PILE)
samples = gen + pl
rng.shuffle(samples)
assert len(samples) == N_GEN + N_PILE and all(s.shape == (1, SEQ) for s in samples), [s.shape for s in samples][:3]
torch.save(samples, out)
used_think = sum(1 for r in think)  # informational
print(f"wrote {len(samples)} samples x {SEQ} tokens ({len(gen)} gen, {len(pl)} pile) -> {out}")
print("decode check:", repr(tok.decode(gen[0][0][:60])))
