"""Synthetic augmentation: Qwen2.5-7B writes extra neutral inputs for every real post.

The target always stays the real post. A candidate input is rejected if the post contains a number or an
entity the input does not support (the model would learn to invent facts), if it leaks style markers, slips
into another language, has the wrong length or duplicates another input. Accepted inputs were then read by hand
(a few were corrected manually).

    python synth.py type1   # -> data/synthetic/type1.jsonl
"""

import argparse

from common import (EMOJI_RE, HASHTAG_RE, MENTION_RE, ROOT, SEED, STYLES, URL_RE, check_entities, check_numbers,
                    clean, cyrillic_ratio, jaccard, load_pairs, load_posts, ngrams, sentences, write_jsonl)
from generate import HF

NEUTRALIZER = "Qwen/Qwen2.5-7B-Instruct"
EXAMPLE = (
    "Правительство Москвы утвердило изменение тарифов на проезд в наземном городском транспорте. "
    "Согласно постановлению, с 1 января стоимость поездки по единому билету увеличится с 62 до 65 "
    "рублей, а по карте «Тройка» — с 54 до 56 рублей. В департаменте транспорта пояснили, что "
    "индексация связана с ростом расходов на обслуживание подвижного состава и оплату труда водителей. "
    "Льготные категории пассажиров сохранят право бесплатного проезда. По данным ведомства, "
    "дополнительные средства будут направлены на обновление автобусного парка."
)
SYSTEM = (
    "Ты — редактор информационного агентства. Тебе дают пост из Telegram-канала и справку по той же "
    "новости. Напиши по ним нейтральную новостную заметку для ленты агентства.\n\n"
    "Требования:\n"
    "- Официально-деловой нейтральный регистр: третье лицо, полные названия организаций и должностей, "
    "без оценок, иронии, шуток, сленга, эмоций, восклицаний.\n"
    "- Без эмодзи, хэштегов, ссылок, @упоминаний, подписей канала и призывов к читателю.\n"
    "- Сохрани ВСЕ факты поста: каждое число, сумму, процент, дату, имя и название должны "
    "присутствовать в заметке. Если число в посте и в справке расходится, используй число из поста.\n"
    "- Не добавляй новых чисел, дат, имён и событий. Можно добавлять только нейтральные пояснения "
    "общего характера и подробности из справки.\n"
    "- Один абзац, около {chars} символов.\n"
    "- Выведи только текст заметки, без заголовка и пояснений.\n\n"
    "Пример регистра и формата (другая новость):\n{example}"
)
TARGET_CHARS = (900, 650)  # test inputs are ~900 chars, the original train inputs ~550


def rejection_reasons(text, post, chars, others):
    missing = check_numbers(text, post["output"])[0]
    _, novel = check_entities(text, post["output"])
    _, novel_orig = check_entities(post["input"], post["output"])
    markers = [name for name, rx in (("emoji", EMOJI_RE), ("hashtag", HASHTAG_RE), ("mention", MENTION_RE),
                                     ("url", URL_RE)) if rx.search(text)]
    reasons = {
        f"post_numbers_missing:{missing}": missing,
        "post_entities_missing": len(novel) > len(novel_orig),
        f"style_markers:{markers}": markers,
        "not_russian": cyrillic_ratio(text) < 0.85,
        f"length:{len(text)}": not 0.6 * chars <= len(text) <= 1.5 * chars,
        "too_few_sentences": len(sentences(text)) < 2,
        "copy_of_post": jaccard(ngrams(text), ngrams(post["output"])) > 0.7,
        "near_duplicate": any(jaccard(ngrams(text), ngrams(o)) >= 0.85 for o in [post["input"], *others]),
    }
    return [r for r, bad in reasons.items() if bad]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("style", choices=STYLES)
    a = p.parse_args()
    gen = HF(NEUTRALIZER, adapters=(), load_in_4bit=True)
    val_ids = {p["post_id"] for p in load_pairs(a.style, synthetic=False)[1]}  # val posts get no variants
    tasks = [(post, v) for post in load_posts(a.style) if post["id"] not in val_ids for v in range(len(TARGET_CHARS))]
    accepted = {}
    for attempt in range(3):
        todo = [t for t in tasks if (t[0]["id"], t[1]) not in accepted]
        chats = [[{"role": "system", "content": SYSTEM.format(chars=TARGET_CHARS[v], example=EXAMPLE)},
                  {"role": "user", "content": f"Пост канала:\n{post['output']}\n\nСправка:\n{post['input']}"}]
                 for post, v in todo]
        outs = gen(chats, None, 3, 512, 0.8, SEED + 101 * attempt, top_p=0.95)
        for (post, v), raws in zip(todo, outs, strict=True):
            for raw, _ in raws:
                text = " ".join(clean(raw, "type2").split())  # strip chat artefacts, one paragraph
                others = [t for (pid, _), t in accepted.items() if pid == post["id"]]
                reasons = rejection_reasons(text, post, TARGET_CHARS[v], others)
                print(post["id"], v, "OK" if not reasons else reasons)
                if not reasons:
                    accepted[(post["id"], v)] = text
                    break
    rows = [{"post_id": pid, "variant": v, "input": t} for (pid, v), t in sorted(accepted.items())]
    write_jsonl(ROOT / f"data/synthetic/{a.style}.jsonl", rows)
    print(f"accepted {len(rows)}/{len(tasks)}")


if __name__ == "__main__":
    main()
