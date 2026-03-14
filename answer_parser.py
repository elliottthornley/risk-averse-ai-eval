#!/usr/bin/env python3
"""Shared permissive answer parser used across evaluation scripts."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata
from typing import List, Optional, Set


_ROMAN_TO_INT = {
    "i": 1,
    "ii": 2,
    "iii": 3,
    "iv": 4,
    "v": 5,
    "vi": 6,
    "vii": 7,
    "viii": 8,
    "ix": 9,
    "x": 10,
}


@dataclass(frozen=True)
class ChoiceParseResult:
    choice: Optional[str]
    strategy: Optional[str]


def _normalize_response_text(response: str) -> str:
    text = unicodedata.normalize("NFKC", response)
    text = text.replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    # Strip common reasoning wrappers while preserving the reasoning text itself.
    text = re.sub(r"</?(?:think|thinking|reasoning|analysis)\b[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<\|[^|]+?\|>", " ", text)
    text = re.sub(r"[*_`]+", "", text)
    return text.rstrip()


def _normalize_label_token(token: str) -> str:
    s = unicodedata.normalize("NFKC", token).strip().lower()
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    s = re.sub(r"^(?:option|answer)\s*", "", s)
    s = s.strip(" \t\r\n'\"`*_[](){}<>.,;:!?")
    if s in _ROMAN_TO_INT:
        s = str(_ROMAN_TO_INT[s])
    return s


def _valid_options(num_options: int) -> Set[str]:
    letters = {chr(ord("a") + i) for i in range(num_options)}
    numbers = {str(i + 1) for i in range(num_options)}
    return letters | numbers


def _extract_valid_matches(pattern: str, text: str, valid: Set[str]) -> List[str]:
    matches = []
    for m in re.finditer(pattern, text, flags=re.IGNORECASE):
        token = _normalize_label_token(m.group(1))
        if token in valid:
            matches.append(token)
    return matches


def _last_valid_match(pattern: str, text: str, valid: Set[str]) -> Optional[str]:
    matches = _extract_valid_matches(pattern, text, valid)
    return matches[-1] if matches else None


def _tail_sentences(text: str, limit: int = 8) -> List[str]:
    chunks = re.split(r"(?<=[.!?])\s+|\n+", text)
    sentences = [chunk.strip() for chunk in chunks if chunk and chunk.strip()]
    return sentences[-limit:]


def extract_json_answer(response: str) -> Optional[str]:
    """Extract answer from JSON snippets like {"answer":"2"}."""
    if not isinstance(response, str) or not response.strip():
        return None
    text = _normalize_response_text(response)
    pattern = r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}'
    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    candidate = None
    for m in matches:
        candidate = _normalize_label_token(m.group(1))
    return candidate


def extract_choice_with_strategy(response: str, num_options: int) -> ChoiceParseResult:
    """
    Extract a model choice using permissive but bounded matching.
    Returns both the parsed choice and the strategy used.
    """
    if not isinstance(response, str) or not response.strip():
        return ChoiceParseResult(choice=None, strategy=None)

    text = _normalize_response_text(response).lower()
    tail = text[-3000:] if len(text) > 3000 else text
    valid = _valid_options(num_options)

    # 1) JSON answer format (most explicit).
    json_choice = _last_valid_match(
        r'\{\s*["\']answer["\']\s*:\s*["\']?\s*([a-z0-9ivx]+)\s*["\']?\s*\}',
        text,
        valid,
    )
    if json_choice:
        return ChoiceParseResult(choice=json_choice, strategy="json")

    # 2) Boxed answer format often used by reasoning models.
    boxed_choice = _last_valid_match(r'\\boxed\s*\{\s*([a-z0-9ivx]+)\s*\}', text, valid)
    if boxed_choice:
        return ChoiceParseResult(choice=boxed_choice, strategy="boxed")

    # 3) Explicit answer markers.
    answer_choice = _last_valid_match(
        r'(?:final\s+answer|final|answer|my\s+answer|choice)\s*[:\-]?\s*'
        r'(?:is\s+)?(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?'
        r'(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))',
        tail,
        valid,
    )
    if answer_choice:
        return ChoiceParseResult(choice=answer_choice, strategy="answer_marker")

    # 4) Decision verbs.
    decision_choice = _last_valid_match(
        r"(?:i(?:'d)?\s+)?(?:would\s+)?(?:choose|select|pick|chose|selected|picking|"
        r"opt\s+for|go\s+with|prefer|recommend|suggest)\s+(?:option\s*)?[\(\[]?\s*"
        r"([a-z0-9ivx]+)\s*[\)\]]?(?=\s*(?:$|[\n\r\.\,\;\:\!\)]|\b(?:because|as|since|for)\b))",
        tail,
        valid,
    )
    if decision_choice:
        return ChoiceParseResult(choice=decision_choice, strategy="decision_verb")

    # 5) Conclusive statement about best/most attractive option.
    option_is_choice = _last_valid_match(
        r'\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\s+'
        r'(?:is|seems|looks|appears|has)\s+(?:the\s+)?'
        r'(?:best|better|preferred|preferable|optimal|most\s+attractive|'
        r'highest\s+expected\s+(?:utility|value))\b',
        tail,
        valid,
    )
    if option_is_choice:
        return ChoiceParseResult(choice=option_is_choice, strategy="option_is_best")

    # 6) Conclusive last-sentence forms often emitted inside thinking-only blocks.
    tail_sentences = _tail_sentences(tail)

    best_choice_patterns = [
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*(?:the\s+)?'
            r'(?:best|preferred|correct|right|optimal|most\s+attractive)\s+'
            r'(?:option|choice|answer)\s*(?:is|would\s+be|seems\s+to\s+be)\s*'
            r'(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?',
            "best_choice_is",
        ),
        (
            r'(?:therefore|thus|so|hence|overall|ultimately)?\s*,?\s*'
            r'(?:i\s+)?(?:should|would|will|must|ought\s+to)\s+'
            r'(?:choose|select|pick|go\s+with|opt\s+for)\s+'
            r'(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?',
            "decision_modal",
        ),
    ]
    for sentence in reversed(tail_sentences):
        for pattern, strategy in best_choice_patterns:
            choice = _last_valid_match(pattern, sentence, valid)
            if choice:
                return ChoiceParseResult(choice=choice, strategy=strategy)

    lines = [line.strip() for line in tail.splitlines() if line.strip()]

    # 7) Short explicit answer lines near the end.
    for line in reversed(lines[-8:]):
        if len(line) > 90:
            continue
        m = re.fullmatch(
            r'(?:final\s+answer|final|answer|choice)?\s*[:\-]?\s*(?:is\s+)?'
            r'(?:option\s*)?[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\.?',
            line,
            flags=re.IGNORECASE,
        )
        if not m:
            continue
        token = _normalize_label_token(m.group(1))
        if token in valid:
            return ChoiceParseResult(choice=token, strategy="short_answer_line")

    # 8) If the entire response is just the option token.
    compact = re.sub(r"\s+", "", text)
    compact = _normalize_label_token(compact)
    if compact in valid:
        return ChoiceParseResult(choice=compact, strategy="bare_token")

    # 9) Last-line fallback for terse outputs like "Option (2).".
    tail_lines = "\n".join(lines[-3:])
    fallback_hits = _extract_valid_matches(
        r"\boption\s*[\(\[]?\s*([a-z0-9ivx]+)\s*[\)\]]?\b",
        tail_lines,
        valid,
    )
    if fallback_hits:
        unique = set(fallback_hits)
        if len(unique) == 1:
            return ChoiceParseResult(choice=fallback_hits[-1], strategy="tail_option_fallback")

    return ChoiceParseResult(choice=None, strategy=None)


def extract_choice_permissive(response: str, num_options: int) -> Optional[str]:
    """Compatibility wrapper returning just the parsed choice."""
    return extract_choice_with_strategy(response, num_options).choice
