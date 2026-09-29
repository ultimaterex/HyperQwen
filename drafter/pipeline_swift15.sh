#!/bin/bash
# Swift-1.5 uncensored: capture -> own-output draft vocab -> MTP Hessians -> GPTQ int4 lm_head -> draft head -> GPTQ int4 MTP
# -> integrity check. Starts when drafter/gen_data.py has finished writing $DD/gen.jsonl.
set -e
cd /home/rex/ai/qwen38-27b-rtx3090
export CUDA_HOME=/usr/local/cuda-13.4 PATH=/usr/local/cuda-13.4/bin:/home/rex/ai/qwen38-27b-rtx3090/venv/bin:$PATH CUDA_VISIBLE_DEVICES=0
V=venv/bin/python
M=models/Swift-1.5-Qwen3.8-27B-Uncensored-W4A16-AutoRound
T=models/tmp-swift15-lm4
F=models/Swift-1.5-Qwen3.8-27B-Uncensored-W4A16-AutoRound-fast
DD=/mnt/nvme1/models_archive/sources/swift15-distill
log() { echo "=== $(date +%T) $*"; }

log "waiting for gen_data to finish"
until grep -q "^done$" $DD/gen.log && ! pgrep -f "drafter/gen_data.py" >/dev/null; do sleep 60; done
log "gen done: $(wc -l < $DD/gen.jsonl) records"

log "capture (hidden states of every token)"
DRAFT_DATA=$DD MODEL=$M $V drafter/capture.py 2>&1 | grep -vE "min_frames|max_frames" | tail -5

log "draft vocab from own outputs"
$V drafter/count_vocab_from_gen.py $M --gen $DD/gen.jsonl --out $DD/draft_vocab_ids_swift15.json 2>&1 | tail -6

log "MTP eval (depths 2) + GPTQ Hessians"
mkdir -p $DD/runs/e
$V drafter/train_mtp.py --model $M --data $DD --out $DD/runs/e --eval-only 1 --draft-ids $DD/draft_vocab_ids_swift15.json \
   --max-seqs 400 --val-frac 0.4 --depths 2 --dump-hessians $DD/runs/e/mtp_hessians.pt 2>&1 | tail -12

log "GPTQ int4 lm_head"
rm -rf $T
DRAFT_DATA=$DD $V drafter/gptq_lm_head.py $M $T --bits 4 --calib-rows 300000 2>&1 | tail -6

log "draft head from the int4 lm_head, own-output vocab"
$V prepare/build_draft_vocab.py $T --ids $DD/draft_vocab_ids_swift15.json 2>&1 | tail -3

log "GPTQ int4 MTP"
rm -rf $F
$V drafter/requant_mtp_gptq.py $T $F $DD/runs/e/mtp_hessians.pt --bits 4 --orig $M 2>&1 | tail -10

log "integrity check"
python3 - "$F" <<'P'
import json, struct, sys, os, collections
D = sys.argv[1]
def keys(f):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]; return [k for k in json.loads(fh.read(n)) if k != "__metadata__"]
wm = json.load(open(f"{D}/model.safetensors.index.json"))["weight_map"]
where = collections.defaultdict(list)
for f in sorted(os.listdir(D)):
    if f.endswith(".safetensors"):
        for k in keys(f"{D}/{f}"): where[k].append(f)
bad = {"dups": sum(len(v) > 1 for v in where.values()), "not_indexed": sum(k not in wm for k in where),
       "wrong_file": sum(k in wm and wm[k] not in v for k, v in where.items()), "missing": sum(k not in where for k in wm)}
print("tensors", len(where), bad)
q = json.load(open(f"{D}/config.json"))["quantization_config"]["config_groups"]
print({k: (v["weights"]["num_bits"], v["targets"]) for k, v in q.items()})
sys.exit(1 if any(bad.values()) else 0)
P
log "PIPELINE DONE"
