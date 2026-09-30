#!/bin/bash
# harder T2 (10 cycles x 16 ops, 24k-token chunks) on A and B, production -2x config
cd /home/rex/ai/qwen38-27b-rtx3090
DC="docker compose -f docker-compose.yml --project-directory . --profile single"
for pair in "A:Qwen3.8-27B-W4A16-AutoRound-swift-unc-gptq-full-mtpfix" "B:Qwen3.8-27B-W4A16-AutoRound-swift-unc-gptq-full-mtpfix-bf16lmhead"; do
  T=${pair%%:*}; M=${pair#*:}; ML=196608; [ "$T" = B ] && ML=153600   # bf16 lm_head: max 153,920 on 24 GB
  E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=mtp CTX=long MAX_LEN=$ML PREFIX_CACHE=1 INT8_ACT=int8 VISION=1 KV_OFFLOAD_GB=24 MAX_SEQS=2 MODEL=/app/models/$M"
  echo "=== hard $T boot $(date +%T)"
  env $E $DC up -d >/dev/null 2>&1
  for i in $(seq 1 240); do curl -sf localhost:18020/health >/dev/null && break; sleep 5; done
  venv/bin/python bench/longctx_suite/t2_compaction.py ${T}-hard --hard
  env $E $DC down >/dev/null 2>&1; /home/rex/ai/llama-swap/syv-stop.sh single >/dev/null 2>&1
done
echo "HARD DONE"
