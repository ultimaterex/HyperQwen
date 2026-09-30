#!/bin/bash
# usage: run_suite.sh <phase> <tag> <model_dir_name> [extra env...]
#   phase t1   -> SPEC=off CTX=long MAX_LEN=70000 PREFIX_CACHE=0 VISION=0 (prompt_logprobs headroom)
#   phase t23  -> production -2x config (SPEC=mtp CTX=long MAX_LEN=196608 ...), runs T2 then T3
cd /home/rex/ai/qwen38-27b-rtx3090
PH=$1; TAG=$2; M=$3; shift 3
DC="docker compose -f docker-compose.yml --project-directory . --profile single"
if [ "$PH" = t1 ]; then
  E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=off CTX=long MAX_LEN=70000 PREFIX_CACHE=0 VISION=0 MAX_SEQS=1 MODEL=/app/models/$M $*"
else
  E="PORT=18020 CUDA_VISIBLE_DEVICES=0 SPEC=mtp CTX=long MAX_LEN=196608 PREFIX_CACHE=1 INT8_ACT=int8 VISION=1 KV_OFFLOAD_GB=24 MAX_SEQS=2 MODEL=/app/models/$M $*"
fi
echo "=== $PH $TAG boot $(date +%T)  [$E]"
env $E $DC up -d >/dev/null 2>&1
for i in $(seq 1 240); do curl -sf localhost:18020/health >/dev/null && break; docker ps -a --format '{{.Names}} {{.Status}}' | grep -q "single-1 Exited" && break; sleep 5; done
if ! curl -sf localhost:18020/health >/dev/null; then echo "=== $TAG BOOT FAILED"; docker logs qwen38-27b-rtx3090-single-1 2>&1 | grep -aE "Error|error" | tail -3; env $E $DC down >/dev/null 2>&1; exit 1; fi
docker logs qwen38-27b-rtx3090-single-1 2>&1 | grep -a "GPU KV cache size" | tail -1 | cut -c1-160
if [ "$PH" = t1 ]; then
  venv/bin/python bench/longctx_suite/t1_longppl.py $TAG
else
  venv/bin/python bench/longctx_suite/t2_compaction.py $TAG
  venv/bin/python bench/longctx_suite/t3_math_longctx.py $TAG
fi
env $E $DC down >/dev/null 2>&1
/home/rex/ai/llama-swap/syv-stop.sh single >/dev/null 2>&1
echo "=== $PH $TAG done $(date +%T)"
