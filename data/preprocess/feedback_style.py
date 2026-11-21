import re


def sentence_count(answer: str) -> int:
    return len([part for part in re.split(r"(?<=[.!?])\s+", answer.strip()) if part])


FIRST_PERSON = re.compile(r"\b(?:I|me|my|mine|myself|we|us|our|ours|ourselves)\b", re.I)
SECOND_PERSON = re.compile(r"\b(?:you|your|yours|yourself|yourselves)\b", re.I)


IMPERATIVE = re.compile(
    r"(?:^|[.!?]\s+)(?:(?:please|next time|however)[,\s]+)*"
    r"(?:(?:after|during|when|while|before)\b[^.!?;]+?,\s+)?"
    r"(?:try|keep|make sure|maintain|ensure|avoid|remember|focus|use|adjust|bend|"
    r"raise|lower|lift|straighten|extend|relax|tighten|place|push|pull|move|open|"
    r"close|start|stop|land|improve|increase|decrease|turn|spread|control|engage|"
    r"hold|aim|consider|work|stay|pay attention|follow|return|do not)\s+"
    r"(?!(?:and|or)\b)",
    re.I,
)
DIRECT_ADVICE = re.compile(
    r"[,;]\s+(?:(?:but|so|and)\s+)?(?:(?:next time|also)\s+)*"
    r"(?:try to|pay attention to|keep up the good work|keep practicing|do not)\b",
    re.I,
)


def feedback_person_phrase(answer: str) -> str:
    first = FIRST_PERSON.search(answer) is not None
    second = any(pattern.search(answer.strip()) is not None
                 for pattern in (SECOND_PERSON, IMPERATIVE, DIRECT_ADVICE))
    if first and second:
        return "using first-person narration and second-person advice"
    if first:
        return "in the first person"
    if second:
        return "in the second person"
    return "in the third person"
