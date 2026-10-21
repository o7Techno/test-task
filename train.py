"""QLoRA SFT of one style adapter (TRL). Loss only on the post + <|im_end|>.

    python train.py type1                 # -> adapters/type1, reports/train_type1.json
    python train.py type1 --no-synthetic  # ablation -> adapters/type1_nosyn
"""

import argparse
import json
import time

import torch
from datasets import Dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from common import MODEL, ROOT, SEED, STYLES, load_pairs, messages


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("style", choices=STYLES)
    p.add_argument("--no-synthetic", action="store_true")
    a = p.parse_args()
    name = a.style + "_nosyn" * a.no_synthetic
    train, val = load_pairs(a.style, synthetic=not a.no_synthetic)
    tok = AutoTokenizer.from_pretrained(MODEL)

    def rows(pairs):  # prompt rendered exactly as at inference (HF and vLLM use the same chat template)
        return Dataset.from_list([{
            "prompt": tok.apply_chat_template(messages(a.style, x["input"]), tokenize=False, add_generation_prompt=True),
            "completion": x["output"] + "<|im_end|>"} for x in pairs])

    train_ds, val_ds = rows(train), rows(val)
    # 4-bit NF4 base on GPUs < 16 GB (QLoRA), plain LoRA otherwise.
    small_gpu = torch.cuda.get_device_properties(0).total_memory < 16 * 1024**3
    bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if bf16 else torch.float16
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=dtype) if small_gpu else None
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype, device_map={"": 0}, quantization_config=quant)
    model.config.use_cache = False
    if small_gpu:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    # No embeddings / lm_head / modules_to_save: the adapter stays servable by vLLM multi-LoRA.
    lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    args = SFTConfig(
        output_dir=str(ROOT / "checkpoints" / name), num_train_epochs=3, learning_rate=2e-4,
        lr_scheduler_type="cosine", warmup_steps=0.1, per_device_train_batch_size=4, gradient_accumulation_steps=2,
        per_device_eval_batch_size=4, max_length=1024, packing=False, completion_only_loss=True,
        gradient_checkpointing=True, bf16=bf16, fp16=not bf16, optim="adamw_torch", logging_steps=1,
        report_to="none", eval_strategy="epoch", save_strategy="epoch", load_best_model_at_end=True,
        metric_for_best_model="eval_loss", save_only_model=True, seed=SEED, data_seed=SEED,
        dataloader_pin_memory=False,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=lora)

    for i in range(3):  # tokens that receive loss must decode exactly to the completion
        batch = trainer.data_collator([trainer.train_dataset[i]])
        trained = tok.decode(batch["input_ids"][0][batch["labels"][0] != -100])
        assert trained == train_ds[i]["completion"], f"label mask mismatch: {trained!r}"

    start = time.time()
    trainer.train()
    trainer.save_model(str(ROOT / "adapters" / name))  # best epoch by val loss
    evals = [h for h in trainer.state.log_history if "eval_loss" in h]
    report = {
        "train_pairs": len(train), "val_pairs": len(val),
        "eval_loss_by_epoch": [round(h["eval_loss"], 3) for h in evals],
        "best_epoch": min(evals, key=lambda h: h["eval_loss"])["epoch"],
        "minutes": round((time.time() - start) / 60, 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2), "qlora": small_gpu,
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / f"reports/train_{name}.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
