#!/usr/bin/env bash
# Base model + both style adapters behind one vLLM OpenAI-compatible server (1 GPU 24 GB, <=15 requests,
# prompt + answer <= 4096 tokens). The style is chosen by the request's `model` field: type1 | type2.
# GPU_MEMORY_UTILIZATION / MAX_NUM_BATCHED_TOKENS can be overridden; extra flags are passed through ("$@").
set -euo pipefail
: "${VLLM_API_KEY:?set VLLM_API_KEY}"
ADAPTERS_DIR="${ADAPTERS_DIR:-$(dirname "$0")/../adapters}"

exec python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-3B-Instruct \
  --served-model-name qwen2.5-3b-base \
  --dtype bfloat16 \
  --enable-lora \
  --lora-modules "type1=${ADAPTERS_DIR}/type1" "type2=${ADAPTERS_DIR}/type2" \
  --max-loras 2 \
  --max-lora-rank 16 \
  --max-model-len 4096 \
  --max-num-seqs 16 \
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-8192}" \
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.90}" \
  --enable-prefix-caching \
  --host 0.0.0.0 --port 8000 \
  --api-key "${VLLM_API_KEY}" \
  "$@"
