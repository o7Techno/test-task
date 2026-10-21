"""VRAM budget of a vLLM server with multi-LoRA, from the model's config.json.

    python deploy/memory_calc.py                                   # Qwen2.5-3B, 24 GiB, 0.90
    python deploy/memory_calc.py --model Qwen/Qwen2.5-7B-Instruct --gpu-gib 22.5

Weights, LoRA slots and KV are exact; activations, CUDA graphs and the CUDA context are estimates
(vLLM prints the measured values at start-up).
"""

import argparse
import json
import math
from pathlib import Path

GIB = 1024**3


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--gpu-gib", type=float, default=24.0)
    p.add_argument("--util", type=float, default=0.90)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-num-seqs", type=int, default=16)
    p.add_argument("--max-num-batched-tokens", type=int, default=8192)
    p.add_argument("--max-loras", type=int, default=2)
    p.add_argument("--max-lora-rank", type=int, default=16)
    a = p.parse_args()
    from huggingface_hub import hf_hub_download

    c = json.loads(Path(hf_hub_download(a.model, "config.json")).read_text())
    h, layers, inter, vocab = c["hidden_size"], c["num_hidden_layers"], c["intermediate_size"], c["vocab_size"]
    heads, kv_heads = c["num_attention_heads"], c["num_key_value_heads"]
    hd = h // heads
    q, kv = heads * hd, kv_heads * hd
    dims = [(h, q), (h, kv), (h, kv), (q, h), (h, inter), (h, inter), (inter, h)]  # q k v o gate up down
    per_layer = sum(i * o for i, o in dims) + q + 2 * kv + 2 * h  # + qkv biases (Qwen2) + 2 RMSNorms
    params = layers * per_layer + vocab * h * (1 if c.get("tie_word_embeddings") else 2) + h

    weights = 2 * params  # bf16
    lora = 2 * a.max_loras * layers * sum(a.max_lora_rank * (i + o) for i, o in dims)
    act = a.max_num_batched_tokens * 2 * (2 * inter + 4 * h + q + 2 * kv) + a.max_num_seqs * vocab * 4  # est.
    other = 1.1 * GIB  # CUDA graphs ~0.5 + CUDA context / workspaces ~0.6 (est.)
    kv_token = 2 * layers * kv_heads * hd * 2  # K and V, bf16
    kv_need = a.max_num_seqs * a.max_model_len * kv_token
    kv_free = a.util * a.gpu_gib * GIB - weights - lora - act - other
    min_util = (weights + lora + act + other + 1.1 * kv_need) / (a.gpu_gib * GIB)

    print(f"{a.model}: {params / 1e9:.2f}B params, {layers} layers, {kv_heads} KV heads x {hd}")
    print(f"weights + LoRA slots          {(weights + lora) / GIB:6.2f} GiB")
    print(f"activations + graphs + ctx    {(act + other) / GIB:6.2f} GiB  (estimate)")
    print(f"left for KV at util {a.util:.2f}     {kv_free / GIB:6.2f} GiB")
    print(f"KV per token / {a.max_num_seqs} x {a.max_model_len}     {kv_token / 1024:.0f} KiB / {kv_need / GIB:.2f} GiB")
    print(f"max concurrency at {a.max_model_len}     {kv_free / (a.max_model_len * kv_token):6.1f}x")
    print(f"minimal util (KV x1.1)        {math.ceil(min_util * 100) / 100:6.2f}")


if __name__ == "__main__":
    main()
