"""Inspectable NLP preprocessing. Raw offsets always refer to the original text."""
import hashlib
import re
import unicodedata
from collections import Counter

PREPROCESSING_VERSION = "unicode-math-spans-v3"

STOP_WORDS = set("a an the and or but if of to in on at for from with by as is are was were be been it this that these those i you we they he she my your our their have has had do does did can could would should will what how why which who not".split())


def normalize(text):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def text_hash(text):
    return hashlib.sha256(normalize(text).casefold().encode()).hexdigest()


def analyze(text):
    spans = [{"token": m.group(), "start": m.start(), "end": m.end()} for m in re.finditer(r"\d+(?:\.\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)?|[^\w\s]|[\w]", text, re.UNICODE)]
    tokens = [unicodedata.normalize("NFKC", item["token"]).casefold() for item in spans]
    keywords = [word for word, _ in Counter(t for t in tokens if len(t) > 2 and t not in STOP_WORDS).most_common(16)]
    sentences = [{"text": m.group(), "start": m.start(), "end": m.end()} for m in re.finditer(r"[^\n]+?(?:[.!?]+(?=\s|$)|(?=\n)|$)", text) if m.group().strip()]
    # Match common typographic apostrophes in the lexical view only. Source
    # offsets, exact-quote validation and response identity keep the raw text.
    lexical = normalize(text).casefold().replace('’', "'").replace('‘', "'")
    return {"raw_text": text, "normalized_text": normalize(text), "token_spans": spans, "token_count": len(tokens),
            "sentences": sentences, "keywords": keywords, "response_hash": hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest(),
            "negation_cues": re.findall(r"\b(?:not|never|cannot|don't|didn't|isn't|can't|won't|wasn't|aren't|doesn't|couldn't|wouldn't|shouldn't)\b", lexical),
            "uncertainty_cues": re.findall(r"\b(?:guess|unsure|maybe|not sure|don't know|do not know)\b", lexical),
            "version": PREPROCESSING_VERSION}


def chunk_text(text, size=850, overlap=140):
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + size // 2, end)
            if boundary > start:
                end = boundary
        yield {"content": text[start:end], "start": start, "end": end}
        if end == len(text):
            break
        start = max(start + 1, end - overlap)


def search_terms(text):
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in analyze(text)["keywords"][:14])
