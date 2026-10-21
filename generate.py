"""Generate posts: N candidates -> clean + validate -> best by score -> retries -> fallback.

    python generate.py                                  # all systems, local transformers + LoRA
    python generate.py --systems lora lora_fewshot --backend vllm --out outputs/vllm
"""

import argparse
import json
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext

from common import (MODEL, ROOT, STYLES, check, clean, load_pairs, load_test, messages, naive_messages,
                    read_jsonl, select_demos, write_jsonl)

SYSTEMS = ("naive", "fewshot", "lora", "lora_fewshot")
SAMPLING = {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "repetition_penalty": 1.05}
N_CANDIDATES, RETRIES, SEED = 4, 2, 42


class HF:
    """transformers + both LoRA adapters on one base model; adapter=None means the base model."""

    def __init__(self, model=MODEL, adapters=STYLES, load_in_4bit=False, token_budget=3500):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Prompt tokens per generate() call (prompt x candidates): more spills out of 8 GB VRAM and gets ~20x slower.
        self.torch, self.token_budget = torch, token_budget
        self.tok = AutoTokenizer.from_pretrained(model)
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16) if load_in_4bit else None
        self.model = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16, device_map={"": 0},
                                                          quantization_config=quant)
        if adapters:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, ROOT / "adapters" / adapters[0], adapter_name=adapters[0])
            for name in adapters[1:]:
                self.model.load_adapter(ROOT / "adapters" / name, adapter_name=name)
        self.model.eval()

    def __call__(self, chats, adapter, n, max_tokens, temperature, seed, **sampling):
        sampling = {**SAMPLING, "temperature": temperature, **sampling}
        if adapter:
            self.model.set_adapter(adapter)
        base_only = adapter is None and hasattr(self.model, "disable_adapter")
        results = []
        with self.model.disable_adapter() if base_only else nullcontext(), self.torch.inference_mode():
            for i, chat in enumerate(chats):
                prompt = self.tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
                step = max(1, min(n, self.token_budget // enc["input_ids"].shape[1]))
                cands = []
                for j in range(0, n, step):
                    self.torch.manual_seed(seed + 100 * i + j)
                    out = self.model.generate(**enc, **sampling, do_sample=True, num_return_sequences=min(step, n - j),
                                              max_new_tokens=max_tokens, pad_token_id=self.tok.eos_token_id)
                    cands += [self._scored(enc["input_ids"][0], seq) for seq in out[:, enc["input_ids"].shape[1] :]]
                results.append(cands)
        return results

    def _scored(self, prompt, seq):
        """(text, mean log-prob of its tokens under the current model), the log-prob measures fluency."""
        ends = self.torch.isin(seq, self.torch.tensor(self.model.generation_config.eos_token_id, device=seq.device))
        seq = seq[: int(ends.nonzero()[0, 0]) + 1] if ends.any() else seq
        logits = self.model(input_ids=self.torch.cat([prompt, seq])[None], logits_to_keep=len(seq) + 1).logits
        logprob = logits[0, :-1].float().log_softmax(-1).gather(1, seq[:, None]).mean().item()
        return self.tok.decode(seq, skip_special_tokens=True), logprob


class VLLM:
    """OpenAI-compatible vLLM server; the LoRA adapter is selected by the `model` field."""

    def __init__(self, base_url=None, api_key=None, base_name="qwen2.5-3b-base"):
        self.url = (base_url or os.environ.get("OPENAI_BASE_URL", "http://localhost:8000/v1")) + "/chat/completions"
        self.key, self.base_name = api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"), base_name

    def _post(self, chat, adapter, n, max_tokens, temperature, seed):
        body = {"model": adapter or self.base_name, "messages": chat, "n": n, "max_tokens": max_tokens,
                "seed": seed, "logprobs": True, **SAMPLING, "temperature": temperature}
        req = urllib.request.Request(self.url, json.dumps(body).encode(), {
            "Content-Type": "application/json", "Authorization": f"Bearer {self.key}"})
        with urllib.request.urlopen(req, timeout=600) as resp:
            choices = json.load(resp)["choices"]
        lps = [[t["logprob"] for t in c["logprobs"]["content"]] or [0.0] for c in choices]
        return [(c["message"]["content"], sum(lp) / len(lp)) for c, lp in zip(choices, lps, strict=True)]

    def __call__(self, chats, adapter, n, max_tokens, temperature, seed):
        with ThreadPoolExecutor(8) as pool:
            return list(pool.map(lambda c: self._post(c, adapter, n, max_tokens, temperature, seed), chats))


def run(gen, system, style, items, pool):
    """One record per input: the best passing candidate, else the one with the fewest errors."""
    if system == "naive":  # "before": one raw sample, no cleaning or validation (judged as is)
        outs = gen([naive_messages(style, it["text"]) for it in items], None, 1, 512, SAMPLING["temperature"], SEED)
        return [{**it, "system": system, "output": o[0][0].strip()} for it, o in zip(items, outs, strict=True)]
    chats = [messages(style, it["text"], select_demos(it["text"], pool) if "fewshot" in system else ())
             for it in items]
    adapter = style if system.startswith("lora") else None
    cands = [[] for _ in items]
    todo = list(range(len(items)))
    for attempt in range(RETRIES + 1):  # each retry samples more conservatively
        outs = gen([chats[i] for i in todo], adapter, N_CANDIDATES, 256, 0.7 - 0.15 * attempt, SEED + 1000 * attempt)
        for i, raws in zip(todo, outs, strict=True):
            for raw, logprob in raws:
                text = clean(raw, style)
                errors, score = check(items[i]["text"], text, style)
                cands[i].append({"text": text, "errors": errors, "score": round(score + logprob, 4),
                                 "logprob": round(logprob, 4), "attempt": attempt})
        todo = [i for i in todo if not any(not c["errors"] for c in cands[i])]
        if not todo:
            break
    records = []
    for it, cs in zip(items, cands, strict=True):
        best = min(cs, key=lambda c: (len(c["errors"]), -c["score"]))
        records.append({**it, "system": system, "output": best["text"], "passed": not best["errors"], "candidates": cs})
    return records


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", nargs="+", choices=SYSTEMS, default=SYSTEMS)
    p.add_argument("--styles", nargs="+", choices=STYLES, default=STYLES)
    p.add_argument("--backend", choices=("hf", "vllm"), default="hf")
    p.add_argument("--out", default="outputs")
    a = p.parse_args()
    gen = HF() if a.backend == "hf" else VLLM()
    out = ROOT / a.out
    txt_dir = ROOT if a.out == "outputs" else out  # the deliverables live in the repo root
    for style in a.styles:
        train, val = load_pairs(style)
        items = [{"id": t["id"], "split": "test", "text": t["text"]} for t in load_test()]
        items += [{"id": v["post_id"], "split": "val", "text": v["input"], "reference": v["output"]} for v in val]
        for system in a.systems:
            records = run(gen, system, style, items, train)
            write_jsonl(out / f"{system}_{style}.jsonl", records)
            print(f"{system}/{style}: done" if system == "naive" else
                  f"{system}/{style}: {sum(r['passed'] for r in records)}/{len(records)} passed")
        if (out / f"lora_{style}.jsonl").exists():  # final = LoRA, failed inputs replaced by LoRA + few-shot
            fallback = {r["id"]: r for r in read_jsonl(out / f"lora_fewshot_{style}.jsonl") if r["passed"]} \
                if (out / f"lora_fewshot_{style}.jsonl").exists() else {}
            final = [r if r["passed"] or r["id"] not in fallback else {**fallback[r["id"]], "fallback": True}
                     for r in read_jsonl(out / f"lora_{style}.jsonl")]
            write_jsonl(out / f"final_{style}.jsonl", final)
            txt = "\n\n---\n\n".join(f"### Input {r['id']}\n\n{r['output']}" for r in final if r["split"] == "test")
            (txt_dir / f"outputs_{style}.txt").write_text(txt + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
