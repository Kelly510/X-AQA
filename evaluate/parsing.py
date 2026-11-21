from __future__ import annotations
import re, json, math
from functools import lru_cache
from pathlib import Path


def user_text(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "user":
            return message.get("content") or ""
    return ""


def gold_text(row: dict) -> str:
    label = row.get("labels")
    if isinstance(label, str) and label.strip():
        return label
    for message in reversed(row.get("messages") or []):
        if message.get("role") == "assistant":
            return message.get("content") or ""
    return ""


THINK_RE = re.compile(r"<think>.*?</think>", re.S)


def pred_text(row: dict) -> str:
    response = row.get("response")
    if isinstance(response, list):
        response = response[0] if response else ""
    response = THINK_RE.sub("", response or "")
    if "<think>" in response:
        response = response.split("<think>", 1)[0]
    elif "</think>" in response:
        response = response.rsplit("</think>", 1)[1]
    return response.replace("</think>", "").strip()


def _row_media(row: dict) -> str:
    for key in ("videos", "images"):
        value = row.get(key)
        if not value:
            continue
        first = value[0] if isinstance(value, (list, tuple)) else value
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            return first.get("path") or ""
    return ""


LLMFMS_HEADER_RE = re.compile(r'movement "([^"]+)" from a single keyframe')

LLMFMS_ITEM_RE = re.compile(r"^(\d+)\. (.+?) -- ", re.M)

LLMFMS_OPTION_RE = re.compile(r"^   - (.+?): ", re.M)

LLMFMS_RATING_RE = re.compile(r"^   (\d+) = ", re.M)

LLMFMS_RANGE_RE = re.compile(r'"Score: <([^>]+)>"')
LLMFMS_ALLOWED_RE = re.compile(r"allowed score values are ([0-9,\sor]+)", re.I)


LLMFMS_SCHEMA_ITEM_RE = re.compile(r'"([^"]+)"\s*:\s*"<(.*?)>"')
LLMFMS_SCHEMA_SCORE_RE = re.compile(r'"Score"\s*:\s*<([^>]*)>')

JSON_FENCE_RE = re.compile(
    r"(?P<fence>`{3,}|~{3,})[ \t]*(?:json)?[ \t]*\r?\n" r"(?P<body>.*?)\r?\n(?P=fence)[ \t]*", re.I | re.S
)


def json_object(text):
    if isinstance(text, str) and (match := JSON_FENCE_RE.fullmatch(text.strip())):
        text = match.group("body")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Duplicate JSON key")
            value[key] = item
        return value

    def invalid_constant(value):
        raise ValueError("Nonfinite JSON number")

    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("Nonfinite JSON number")
        return result

    try:
        value = json.loads(text, object_pairs_hook=pairs, parse_constant=invalid_constant, parse_float=finite_float)
        if isinstance(value, dict):
            return _drop_reason(value)
    except (TypeError, ValueError, RecursionError):
        pass

    return _recover_json_object(text if isinstance(text, str) else "")


KEY_SPLIT_RE = re.compile(r"[\s_\-]+")


def _norm_key(key):
    return KEY_SPLIT_RE.sub(" ", str(key)).strip().lower()


def canonical_keys(value, declared):
    if not isinstance(value, dict) or not declared:
        return value
    wanted = {_norm_key(name): name for name in declared}
    if len(wanted) != len(tuple(declared)):
        return value
    for _ in range(2):
        if any(_norm_key(k) in wanted for k in value):
            break
        if len(value) != 1:
            break
        inner = next(iter(value.values()))
        if not isinstance(inner, dict) or not inner:
            break
        value = _drop_reason(inner)
    renamed, seen = {}, set()
    for key, item in value.items():
        target = wanted.get(_norm_key(key), key)
        if target in seen:
            return value
        seen.add(target)
        renamed[target] = item
    return renamed


def _drop_reason(value):
    return {k: v for k, v in value.items() if k != "reason"} if "reason" in value else value


def _recover_json_object(text):
    candidates = [m.group("body") for m in JSON_FENCE_RE.finditer(text)]
    starts = [m.start() for m in re.finditer(r"\{", text)]
    for start in reversed(starts):
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    candidates.append(text[start : end + 1])
                    break
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except (TypeError, ValueError, RecursionError):
            continue
        if isinstance(value, dict):
            return _drop_reason(value)
    return None


def system_text(row: dict) -> str:
    for message in row.get("messages") or []:
        if message.get("role") == "system":
            return message.get("content") or ""
    return ""


def _llmfms_schema(system: str):
    tail = system.partition("Return only this JSON object")[2] or system
    items = []
    for match in LLMFMS_SCHEMA_ITEM_RE.finditer(tail):
        name, body = match.group(1).strip(), match.group(2)
        if name == "Score":
            continue
        options = [option.strip() for option in body.split("|") if option.strip()]
        if not options:
            return None
        items.append((len(items) + 1, name, options))
    found = LLMFMS_SCHEMA_SCORE_RE.search(tail)
    scores = [int(token) for token in re.findall(r"\d+", found.group(1))] if found else []
    return (items, scores) if items and scores else None


@lru_cache(maxsize=128)
def _llmfms_rubric(system: str):
    header = LLMFMS_HEADER_RE.search(system)
    if not header:
        return None
    starts = [(match.start(), int(match.group(1)), match.group(2).strip()) for match in LLMFMS_ITEM_RE.finditer(system)]
    if not starts:
        schema = _llmfms_schema(system)
        return None if schema is None else (header.group(1), *schema)
    bounds = [pos for pos, _, _ in starts] + [len(system)]
    items = []
    for slot, (_, index, name) in enumerate(starts):
        block = system[bounds[slot] : bounds[slot + 1]]
        options = [match.group(1) for match in LLMFMS_OPTION_RE.finditer(block)]
        if not options:
            return None
        items.append((index, name, options))
    rating_block = system.partition("Overall score:\n")[2].split("Output format:", 1)[0]
    scores = [int(token) for token in LLMFMS_RATING_RE.findall(rating_block)]
    if not scores and (found := LLMFMS_RANGE_RE.search(system)):
        scores = [int(token) for token in re.findall(r"\d", found.group(1))]
    elif not scores and (found := LLMFMS_ALLOWED_RE.search(system)):
        scores = [int(token) for token in re.findall(r"\d+", found.group(1))]
    if not scores and (schema := _llmfms_schema(system)):
        scores = schema[1]
    return header.group(1), items, scores


def _llmfms_answer(text: str, items: list):
    value = json_object(text)
    if value is None:
        return {index: None for index, _, _ in items}, None
    expected = {name for _, name, _ in items} | {"Score"}
    if set(value) != expected:
        return {index: None for index, _, _ in items}, None
    verdicts = {}
    for index, name, options in items:
        selected = value.get(name)
        verdicts[index] = selected if isinstance(selected, str) and selected in options else None
    score = value.get("Score")
    if isinstance(score, bool) or not isinstance(score, int):
        score = None
    return verdicts, score


def _benchmark_key(row):
    kw = row.get("chat_template_kwargs") or {}
    return (Path(_row_media(row)).name, system_text(row), user_text(row), kw.get("video_start"), kw.get("video_end"))


def _bounded_number(value, low, high):
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and low <= number <= high else None
