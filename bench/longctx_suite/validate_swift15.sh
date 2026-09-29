#!/bin/bash
# Head-to-head validation of the Swift-1.5 uncensored build (S15) against v1 swift-unc (published, results "A").
cd /home/rex/ai/qwen38-27b-rtx3090
M=Swift-1.5-Qwen3.8-27B-Uncensored-W4A16-AutoRound-fast
DC="docker compose -f docker-compose.yml --project-directory . --profile single"
S=/tmp/claude-1000/-home-rex-ai/753fb34e-05c5-4230-bf60-79932ab7e696/scratchpad
boot() {  # $1 = env string
  env $1 $DC up -d >/dev/null 2>&1
  for i in $(seq 1 240); do curl -sf localhost:18020/health >/dev/null && return 0; docker ps -a --format '{{.Names}} {{.Status}}' | grep -q "single-1 Exited" && break; sleep 5; done
  echo "BOOT FAILED"; docker logs qwen38-27b-rtx3090-single-1 2>&1 | grep -aE "Error|error" | tail -3; return 1
}
down() { env $1 $DC down >/dev/null 2>&1; /home/rex/ai/llama-swap/syv-stop.sh single >/dev/null 2>&1; }

echo "=== 1. quality battery (SPEC=off CTX=fast) $(date +%T)"
E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=off CTX=fast MODEL=/app/models/$M"
boot "$E" && VLLM_API=http://127.0.0.1:18020/v1 venv/bin/python bench/quality_battery.py s15_fast 2>&1 | grep "PPL\|GSM8K"
down "$E"

E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=mtp CTX=long MAX_LEN=196608 PREFIX_CACHE=1 INT8_ACT=int8 VISION=1 KV_OFFLOAD_GB=24 MAX_SEQS=2 MODEL=/app/models/$M"
echo "=== 2+3. refusal + -2x workload bench $(date +%T)"
if boot "$E"; then
  docker logs qwen38-27b-rtx3090-single-1 2>&1 | grep -a "GPU KV cache size" | tail -1 | cut -c1-150
  venv/bin/python bench/longctx_suite/refusal_eval.py s15 2>&1 | tail -1
  SIZES="32000 96000 180000" /usr/bin/python3 $S/bench2x.py S15_2x 2>&1 | grep -v "^{" | tail -2
  docker logs qwen38-27b-rtx3090-single-1 2>&1 | grep -ao "Mean acceptance length: [0-9.]*" | awk '{s+=$4;n++} END{printf "MTP mean acceptance length over run: %.2f (%d intervals)\n", s/n, n}'
fi
down "$E"

echo "=== 4. long-context suite $(date +%T)"
bench/longctx_suite/run_suite.sh t1 S15 $M
bench/longctx_suite/run_suite.sh t23 S15 $M
E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=mtp CTX=long MAX_LEN=196608 PREFIX_CACHE=1 INT8_ACT=int8 VISION=1 KV_OFFLOAD_GB=24 MAX_SEQS=2 MODEL=/app/models/$M"
boot "$E" && venv/bin/python bench/longctx_suite/t2_compaction.py S15-hard --hard 2>&1 | tail -1
down "$E"
echo "=== VALIDATION DONE $(date +%T)"
