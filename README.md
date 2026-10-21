# Нейтральная новость в пост Telegram-канала

`Qwen/Qwen2.5-3B-Instruct` + два LoRA-адаптера: **type1** (ирония про финансы, @banki_oil) и **type2** (городской
новостник, @moscowach). Оба адаптера обслуживает один vLLM-сервер, стиль выбирается полем `model`.

- Выводы: [outputs_type1.txt](outputs_type1.txt), [outputs_type2.txt](outputs_type2.txt)
- Отчёт (архитектура, гиперпараметры, до/после, vLLM): [RESULTS.md](RESULTS.md)

## Запуск

```bash
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python synth.py type1; python synth.py type2   # необязательно: готовая синтетика лежит в data/synthetic/
python train.py type1; python train.py type2   # -> adapters/<style>, reports/train_<style>.json
python generate.py                             # transformers; -> outputs/*.jsonl, outputs_type{1,2}.txt
python evaluate.py                             # -> reports/eval.md
```

## Сервинг

```bash
pip install vllm==0.32.0
VLLM_API_KEY=secret bash deploy/serve_vllm.sh
OPENAI_API_KEY=secret python generate.py --backend vllm   # так получены выводы в репозитории
python deploy/load_test.py --api-key secret --concurrency 15 30
python deploy/memory_calc.py            # бюджет VRAM для 24 GB
```

## Файлы

| Файл | Что делает |
|---|---|
| `common.py` | данные и сплит, промпты, очистка вывода, валидатор (формат, длина, язык, выдуманные числа) |
| `synth.py` | дополнительные нейтральные входы от Qwen2.5-7B с фильтрами фактов |
| `train.py` | QLoRA SFT одного адаптера, лосс только по ответу |
| `generate.py` | 4 кандидата → валидация → лучший → ретраи → fallback; бэкенды transformers и vLLM |
| `evaluate.py` | сравнение систем: формат, длина, выдуманные числа, chrF |
| `deploy/` | команда vLLM, расчёт памяти, нагрузочный тест |
