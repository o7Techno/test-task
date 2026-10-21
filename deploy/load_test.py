"""Load test of the vLLM server: worst-case requests (~3.5k prompt tokens + 512 forced output tokens),
type1/type2 mix, while /metrics is polled for KV-cache usage, queue and preemptions.

    python deploy/load_test.py --api-key $VLLM_API_KEY --concurrency 15 30
"""

import argparse
import json
import random
import re
import statistics
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from common import load_posts, system_prompt  # noqa: E402

METRICS = ("vllm:kv_cache_usage_perc", "vllm:num_requests_running", "vllm:num_requests_waiting",
           "vllm:num_preemptions_total")


def call(url, key, body=None):
    req = urllib.request.Request(url, json.dumps(body).encode() if body else None,
                                 {"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    with urllib.request.urlopen(req, timeout=1800) as resp:
        return resp.read().decode()


def metrics(base, key):
    text = call(base.removesuffix("/v1") + "/metrics", key)
    out = dict.fromkeys(METRICS, 0.0)
    for name, value in re.findall(r"^(vllm:\w+)(?:\{[^}]*\})?\s+([\d.eE+-]+)$", text, re.M):
        if name in out:
            out[name] += float(value)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base-url", default="http://localhost:8000/v1")
    p.add_argument("--api-key", default="EMPTY")
    p.add_argument("--concurrency", type=int, nargs="+", default=[15, 30])
    p.add_argument("--prompt-chars", type=int, default=9300, help="~3.1k tokens + ~0.4k system prompt")
    a = p.parse_args()
    texts = [x["input"] for s in ("type1", "type2") for x in load_posts(s)]

    def request(i):
        style = ("type1", "type2")[i % 2]
        filler = " ".join(random.sample(texts, len(texts)))[: a.prompt_chars]
        body = {"model": style, "max_tokens": 512, "ignore_eos": True, "temperature": 0.7, "messages": [
            {"role": "system", "content": system_prompt(style)}, {"role": "user", "content": f"[{i}] {filler}"}]}
        start = time.time()
        try:
            usage = json.loads(call(a.base_url + "/chat/completions", a.api_key, body))["usage"]
            return time.time() - start, usage["prompt_tokens"]
        except Exception as exc:  # noqa: BLE001 - every failure counts as an error
            print("error:", exc, file=sys.stderr)
            return None

    for conc in a.concurrency:
        before, peak, done = metrics(a.base_url, a.api_key), dict.fromkeys(METRICS, 0.0), threading.Event()

        def poll():
            while not done.wait(0.5):
                for k, v in metrics(a.base_url, a.api_key).items():
                    peak[k] = max(peak[k], v)

        threading.Thread(target=poll, daemon=True).start()
        with ThreadPoolExecutor(conc) as pool:
            results = list(pool.map(request, range(2 * conc)))
        done.set()
        ok = [r for r in results if r]
        preempt = metrics(a.base_url, a.api_key)["vllm:num_preemptions_total"] - before["vllm:num_preemptions_total"]
        print(json.dumps({
            "concurrency": conc, "requests": len(results), "errors": len(results) - len(ok),
            "prompt_tokens_mean": round(statistics.mean(t for _, t in ok)) if ok else None,
            "latency_p50_s": round(statistics.median(t for t, _ in ok), 1) if ok else None,
            "kv_usage_peak": round(peak["vllm:kv_cache_usage_perc"], 2),
            "running_peak": peak["vllm:num_requests_running"], "waiting_peak": peak["vllm:num_requests_waiting"],
            "preemptions": preempt}))


if __name__ == "__main__":
    main()
