"""Compare systems on test (format, length, invented numbers, validity) and val (chrF to the real post).

    python evaluate.py   # outputs/*.jsonl -> reports/eval.md
"""

import statistics

from sacrebleu.metrics import CHRF

from common import ROOT, STYLES, check, check_numbers, format_errors, read_jsonl

SYSTEMS = {"naive": "base + наивный промпт", "fewshot": "few-shot", "final": "LoRA (финал)",
           "lora_fewshot": "LoRA + few-shot"}


def pct(flags):
    return f"{100 * sum(flags) / len(flags):.0f}%"


def main():
    lines = ["| Стиль | Система | Формат | Длина | Выдум. числа | Valid | chrF (val) |", "|---|---|---|---|---|---|---|"]
    for style in STYLES:
        for system, title in SYSTEMS.items():
            records = read_jsonl(ROOT / f"outputs/{system}_{style}.jsonl")
            test = [r for r in records if r["split"] == "test"]
            val = [r for r in records if r["split"] == "val"]
            fallbacks = sum(r.get("fallback", False) for r in test)
            lines.append("| " + " | ".join([
                style, title + (f" ({fallbacks} через fallback)" if fallbacks else ""),
                pct([not format_errors(style, r["output"]) for r in test]),
                f"{statistics.mean(len(r['output']) for r in test):.0f}",
                pct([bool(check_numbers(r["text"], r["output"])[0]) for r in test]),
                pct([not check(r["text"], r["output"], style)[0] for r in test]),
                f"{CHRF().corpus_score([r['output'] for r in val], [[r['reference'] for r in val]]).score:.1f}",
            ]) + " |")
    report = "\n".join([
        "# Оценка", "",
        "Test — 5 входов на стиль, val — 4 отложенных реальных поста. Формат — доля постов, выполняющих жёсткие "
        "правила канала; Длина — средняя в символах (реальные посты: type1 ≈ 300, type2 ≈ 250); Выдум. числа — "
        "доля постов с числом, которого нет во входе; Valid — доля постов, прошедших весь валидатор как есть.",
        "", *lines, ""])
    (ROOT / "reports/eval.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
