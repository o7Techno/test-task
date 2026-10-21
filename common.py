"""Data, prompts, output cleaning and validation shared by every script."""

import json
import random
import re
import statistics
from functools import cache
from pathlib import Path

ROOT = Path(__file__).parent
MODEL = "Qwen/Qwen2.5-3B-Instruct"
STYLES = ("type1", "type2")
SEED = 42
SIGNATURE = {"type1": "@banki_oil", "type2": None}


# --- Data -----------------------------------------------------------------------------------


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def load_posts(style):
    rows = read_jsonl(ROOT / f"data/raw/train_{style}.jsonl")
    return [{"id": f"{style}-{i:03d}", "input": r["input"], "output": r["output"]} for i, r in enumerate(rows)]


def load_test():
    return [{"id": str(r["id"]), "text": r["text"]} for r in read_jsonl(ROOT / "data/raw/test_inputs.jsonl")]


def load_pairs(style, synthetic=True):
    """(train, val). Split by post: val = 4 real pairs, synthetic variants of val posts are dropped."""
    posts = load_posts(style)
    val_ids = set(random.Random(f"{SEED}-{style}").sample(sorted(p["id"] for p in posts), 4))
    pairs = [{"post_id": p["id"], "variant": -1, "input": p["input"], "output": p["output"]} for p in posts]
    if synthetic:
        out = {p["id"]: p["output"] for p in posts}
        pairs += [{**s, "output": out[s["post_id"]]} for s in read_jsonl(ROOT / f"data/synthetic/{style}.jsonl")]
    pairs.sort(key=lambda p: (p["post_id"], p["variant"]))
    train = [p for p in pairs if p["post_id"] not in val_ids]
    val = [p for p in pairs if p["post_id"] in val_ids and p["variant"] == -1]
    return train, val


@cache
def length_stats(style):
    """Allowed length [p5 * 0.7, p95 * 1.3] and median length of the real posts."""
    lengths = [len(p["output"]) for p in load_posts(style)]
    q = statistics.quantiles(lengths, n=20, method="inclusive")
    return q[0] * 0.7, q[-1] * 1.3, statistics.median(lengths)


# --- Prompts (the same system prompt is used for training, HF inference and vLLM) -------------

STYLE_GUIDES = {
    "type1": (
        "@banki_oil",
        "Финансы, экономика и общество с иронией: короткая новость и саркастичный вывод.",
        [
            "Один абзац из 2–3 предложений, обычно 250–350 символов, без эмодзи, хэштегов и заголовка.",
            "Первое предложение — главный факт, сформулированный хлёстко и разговорно "
            "(«Россияне скупают…», «Зумеры пережили…», «…, заявил эксперт»).",
            "Второе предложение — ключевые детали и контекст из новости.",
            "Последнее предложение — короткая ироничная концовка-вывод без новых фактов и цифр.",
            "В конце через пробел подпись канала @banki_oil.",
        ],
        [
            "Разговорный, слегка саркастичный, но без оскорблений и мата.",
            "Обобщения «россияне», «зумеры», «в Госдуме» вместо длинных официальных названий.",
        ],
    ),
    "type2": (
        "@moscowach",
        "Городской новостник: эмодзи-маркер, заголовок-лид и короткие абзацы с деталями.",
        [
            "Первая строка: один подходящий по смыслу эмодзи, пробел и заголовок-лид — одно предложение "
            "с главной новостью, часто с уточнением через тире.",
            "Затем пустая строка и 1–2 коротких абзаца по 1–3 предложения с деталями.",
            "Общий объём обычно 200–350 символов.",
            "Эмодзи только в начале; без хэштегов, ссылок, подписи канала и markdown-разметки.",
        ],
        [
            "Живой, простой язык для горожан, короткие предложения, можно бытовые советы читателям.",
            "Конкретика: места, цены, проценты, сроки.",
        ],
    ),
}
FACT_RULES = [
    "Все числа, даты, имена, названия и суммы, которые ты упоминаешь, должны точно совпадать "
    "с заметкой; главные факты не теряй, второстепенные детали можно опустить.",
    "Не придумывай цифры, цитаты, ссылки, источники и события, которых нет в заметке.",
    "Пиши только по-русски.",
    "Выведи только текст поста — без пояснений, вступлений и кавычек вокруг.",
]


def system_prompt(style):
    channel, summary, structure, tone = STYLE_GUIDES[style]
    lines = [f"Ты — редактор Telegram-канала {channel}. {summary}",
             "Перепиши новостную заметку пользователя в пост этого канала."]
    for title, items in (("Формат", structure), ("Тон", tone), ("Правила", FACT_RULES)):
        lines += ["", f"{title}:", *(f"- {s}" for s in items)]
    return "\n".join(lines)


def messages(style, text, demos=()):
    """System prompt, optional few-shot demos as previous turns, then the news text."""
    turns = [m for d in demos for m in ({"role": "user", "content": d["input"].strip()},
                                         {"role": "assistant", "content": d["output"]})]
    return [{"role": "system", "content": system_prompt(style)}, *turns, {"role": "user", "content": text.strip()}]


NAIVE = {"type1": "@banki_oil про финансы и экономику", "type2": "@moscowach с городскими новостями Москвы"}


def naive_messages(style, text):
    """The "before" baseline: no system prompt, so Qwen's default one is applied."""
    return [{"role": "user", "content": f"Перепиши этот текст в стиле Telegram-канала {NAIVE[style]}: {text.strip()}"}]


def select_demos(text, pool, k=3):
    """k demos with the most similar input (char 3-gram Jaccard), at most one variant per post."""
    grams = ngrams(text)
    ranked = sorted(pool, key=lambda p: -jaccard(grams, ngrams(p["input"])))
    demos, seen = [], set()
    for p in ranked:
        if p["post_id"] not in seen:
            demos.append(p)
            seen.add(p["post_id"])
    return demos[:k]


# --- Text helpers ---------------------------------------------------------------------------

_EMOJI = "\U0001f000-\U0001faff☀-➿⬀-⯿⌀-⏿‼⁉™ℹ〰〽㊗㊙"
EMOJI_RE = re.compile(rf"[{_EMOJI}][️\U0001f3fb-\U0001f3ff]*(?:‍[{_EMOJI}]️?)*")
URL_RE = re.compile(r"(?:https?://|www\.|t\.me/)\S+", re.I)
MENTION_RE = re.compile(r"(?<![\w@])@[A-Za-z0-9_]{3,}")
HASHTAG_RE = re.compile(r"(?<![\w#])#\w+")
CJK_RE = re.compile(r"[぀-ヿ㐀-䶿一-鿿가-힯]")


def ngrams(text, n=3):
    t = " ".join(text.lower().split())
    return {t[i : i + n] for i in range(len(t) - n + 1)}


def jaccard(a, b):
    return len(a & b) / len(a | b) if a | b else 1.0


def sentences(text):
    return [s for s in re.split(r"(?<=[.!?…])\s+", text.strip()) if s]


def cyrillic_ratio(text):
    letters = re.findall(r"[^\W\d_]", text)
    return len(re.findall(r"[А-Яа-яЁё]", text)) / len(letters) if letters else 0.0


# Numbers: digits with multipliers ("1,5 млн", "100 000") and number words ("тринадцать", "вдвое").
_MULT = {"тыс": 1e3, "млн": 1e6, "миллион": 1e6, "млрд": 1e9, "миллиард": 1e9, "трлн": 1e12, "триллион": 1e12}
_MULT_RE = r"(?:трлн|триллион\w*|млрд|миллиард\w*|млн|миллион\w*|тыс\w*)\.?"
_DIGITS_RE = re.compile(rf"(?<![\w.,])(\d{{1,3}}(?:[   ]\d{{3}})+|\d+)(?:[.,](\d+))?(?:\s*({_MULT_RE}))?", re.I)
_WORDS = {}
for _line in """0.5 полгода полугода половина половину половины половиной
1.5 полтора полторы полутора
2 два две двух двум двумя дважды вдвое
3 три трех трёх трем трём тремя трижды втрое
4 четыре четырех четырёх четырем четырём четырьмя вчетверо
5 пять пяти пятью впятеро
6 шесть шести шестью
7 семь семи
8 восемь восьми восемью
9 девять девяти девятью
10 десять десяти десятью вдесятеро
40 сорок сорока
50 пятьдесят пятидесяти
60 шестьдесят шестидесяти
70 семьдесят семидесяти
80 восемьдесят восьмидесяти
90 девяносто девяноста
100 сто ста
200 двести двухсот
300 триста трехсот трёхсот
400 четыреста четырехсот четырёхсот
500 пятьсот пятисот""".splitlines():
    _value, *_forms = _line.split()
    _WORDS.update(dict.fromkeys(_forms, float(_value)))
_TEENS = dict(zip("одиннадцат двенадцат тринадцат четырнадцат пятнадцат шестнадцат семнадцат восемнадцат "
                  "девятнадцат двадцат тридцат".split(), [*range(11, 21), 30], strict=True))
_ORDINALS = dict(zip("втор трет четверт пят шест седьм восьм девят десят".split(), range(2, 11), strict=True))
_ORD_END = r"(?:ой|ый|ий|ая|ое|ого|ому|ом|ую|ые|ых|ым|ыми|ья|ье|ьи|ьего|ьей|ьему|ьем|ью|ьих|ьим|ьими)"
_WORD_RE = re.compile(
    rf"(?<!\w)({'|'.join(sorted(_WORDS, key=len, reverse=True))}|(?:{'|'.join(_TEENS)})(?:ь|и|ью)"
    rf"|(?:{'|'.join(_ORDINALS)}){_ORD_END})(?:\s+({_MULT_RE}))?(?!\w)",
    re.I,
)


def _mult(suffix):
    return next((m for k, m in _MULT.items() if suffix and suffix.lower().startswith(k)), 1.0)


def _word_value(word):
    w = word.lower()
    stems = {**_TEENS, **_ORDINALS}
    return _WORDS.get(w) or next(v for s, v in sorted(stems.items(), key=lambda kv: -len(kv[0])) if w.startswith(s))


def numbers(text):
    """[(raw mention, {possible values})]."""
    out = []
    for m in _DIGITS_RE.finditer(text):
        base = float(re.sub(r"\D", "", m.group(1)) + (f".{m.group(2)}" if m.group(2) else ""))
        out.append((m.group(0).strip(), {round(base, 4), round(base * _mult(m.group(3)), 4)}))
    prev_end, prev = -10, None
    for m in _WORD_RE.finditer(text):
        v = _word_value(m.group(1))
        values = {v, v * _mult(m.group(2))}
        if prev and prev >= 20 and v < 10 and text[prev_end : m.start()].isspace():  # "двадцать пять"
            values.add(prev + v)
        out.append((m.group(0), values))
        prev_end, prev = m.end(), v
    return out


def _zeros(x):
    return len(str(int(x))) - len(str(int(x)).rstrip("0")) if x == int(x) else -1


def _supported(values, pool):
    """Some value equals a pool value or rounds it ("20%" for 20.4%, "300" for 290)."""
    return any(v == s or (s and abs(v - s) / abs(s) <= 0.05 and _zeros(v) > _zeros(s)) for v in values for s in pool)


def check_numbers(source, output):
    """(output numbers absent from the source, share of source numbers kept in the output)."""
    src, out = numbers(source), numbers(output)
    src_vals = {v for _, vs in src for v in vs}
    out_vals = {v for _, vs in out for v in vs}
    invented = [raw for raw, vs in out if not _supported(vs, src_vals)]
    kept = sum(_supported(out_vals, vs) for _, vs in src)
    return invented, kept / len(src) if src else 1.0


_CAP_RE = re.compile(r"(?<![\w-])([A-ZА-ЯЁ][\w-]*\w)")


def entities(text):
    """Capitalised words not at a sentence start (cheap NER), lowercased."""
    text = EMOJI_RE.sub("", MENTION_RE.sub(" ", URL_RE.sub(" ", text)))
    found = set()
    for m in _CAP_RE.finditer(text):
        before = text[: m.start()].rstrip(" \t\"'«(")
        if (not before or before[-1] in ".!?…:\n") and not (m.group(1).isupper() and len(m.group(1)) > 1):
            continue
        found.add(m.group(1).lower())
    return found


def same_lemma(a, b):
    """Inflection-tolerant match: "яндекса" ~ "яндекс"."""
    n = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return a == b or (min(len(a), len(b)) >= 4 and n >= max(4, min(len(a), len(b)) - 2))


def check_entities(source, output):
    """(share of source entities kept, output entities absent from the source)."""
    src, out = entities(source), entities(output)
    src_words = {w.lower() for w in re.findall(r"[\w-]+", source)}
    kept = sum(any(same_lemma(e, o) for o in out) for e in src)
    novel = sorted(e for e in out if not any(same_lemma(e, s) for s in src | src_words))
    return kept / len(src) if src else 1.0, novel


# --- Cleaning and validation ----------------------------------------------------------------

_PREAMBLE_RE = re.compile(r"^(?:конечно|вот|ниже|разумеется|хорошо|отлично|готово|sure|here)\b", re.I)
_META_RE = re.compile(
    r"как (?:языковая )?модель|как ИИ|я не могу|в стиле (?:telegram-)?канала|переписанн\w+ (?:текст|пост)"
    r"|стилизованн\w+ (?:текст|пост)|исходн\w+ (?:текст|заметк)",
    re.I,
)


def clean(raw, style):
    """Strip chat artefacts and normalise the layout of a completion."""
    t = re.sub(r"<\|[^|>\s]{1,40}\|>", "", raw).replace("\r\n", "\n").strip()
    t = re.sub(r"^```\w*\n?|\n?```$", "", t).strip()
    lines = t.split("\n")
    while lines:  # drop "Вот пост:" / "Переписанный текст:" lines
        head = lines[0].strip(" *#_")
        if head and not (head.endswith(":") and len(head) < 80 or _PREAMBLE_RE.match(head) and len(head) < 40):
            break
        lines.pop(0)
    # Hashtag-only lines and short label lines ("Абзац:", "Заголовок:").
    lines = [ln for ln in lines if not (ln.strip() and not HASHTAG_RE.sub("", ln).strip())
             and not re.fullmatch(r"[\w ]{1,30}:", ln.strip(" *#_"))]
    t = "\n".join(ln.rstrip() for ln in lines).strip()
    if len(t) > 2 and t[0] + t[-1] in ('""', "«»", "“”") and t[-1] not in t[1:-1]:
        t = t[1:-1].strip()
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", URL_RE.sub("", t))
    t = re.sub(r"\n{2,}", "\n\n", re.sub(r"(?<!\n)\n(?!\n)", "\n\n", t))  # paragraphs = blank lines
    t = re.sub(r"[ \t]{2,}", " ", t).strip()
    sig = SIGNATURE[style]
    if sig:  # exactly one signature, at the end
        t = re.sub(rf"\s*{sig}", "", t).strip()
        return f"{t} {sig}" if t else t
    return re.sub(r"(?:\s*@\w{3,})+$", "", t)


def format_errors(style, text):
    """Hard format rules of the channel, derived from the 25 real posts of each style."""
    paras = [p for p in text.split("\n\n") if p.strip()]
    emoji = EMOJI_RE.findall(text)
    errors = ["hashtag"] if HASHTAG_RE.search(text) else []
    if style == "type1":
        errors += ["emoji"] * bool(emoji) + ["paragraphs"] * (len(paras) > 2)
        errors += ["no_signature"] * (not text.endswith(SIGNATURE[style]))
    else:
        errors += ["no_emoji_lead"] * (not EMOJI_RE.match(text) or len(emoji) > 1)
        errors += ["paragraphs"] * (not 2 <= len(paras) <= 3) + ["mention"] * bool(MENTION_RE.search(text))
    return errors


def check(source, text, style):
    """Validate a (cleaned) post. Returns (errors, rerank score)."""
    if not text.strip():
        return ["empty"], 0.0
    errors = format_errors(style, text)
    lo, hi, median = length_stats(style)
    errors += [f"length:{len(text)}"] * (not lo <= len(text) <= hi)
    errors += ["cjk"] * bool(CJK_RE.search(text)) + ["not_russian"] * (cyrillic_ratio(text) < 0.85)
    errors += ["meta_comment"] * bool(_META_RE.search(text))
    body = text.removesuffix(SIGNATURE[style] or "").rstrip()
    errors += ["truncated"] * (body[-1:] not in tuple('.!?…»")') and not EMOJI_RE.search(body[-2:]))
    words = re.findall(r"\w+", text.lower())
    grams = [tuple(words[i : i + 5]) for i in range(len(words) - 4)]
    sents = [s.lower() for s in sentences(text) if len(s) > 15]
    errors += ["repetition"] * (len(set(sents)) < len(sents) or any(grams.count(g) > 2 for g in set(grams)))
    invented, num_recall = check_numbers(source, text)
    errors += [f"invented_numbers:{', '.join(invented)}"] * bool(invented)
    ent_recall, novel = check_entities(source, text)
    score = 0.5 * num_recall + 0.5 * ent_recall - 0.1 * min(len(novel), 3) - 0.5 * abs(len(text) / median - 1)
    return errors, round(score, 4)
