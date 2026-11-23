"""Inspectable NLP preprocessing. Raw offsets always refer to the original text."""
# Hash text into compact fingerprints used for topic lookup and repeated answers.
import hashlib
# Match tokens, sentence boundaries, whitespace and common language cues using regular expressions.
import re
# Keep a normalized Unicode view for search and analysis alongside the original.
import unicodedata
# Count repeated terms so the lexical search can prefer the most frequent ones.
from collections import Counter

# Record which preprocessing rules produced the saved spans and keywords.
PREPROCESSING_VERSION = "unicode-math-spans-v3"

# Remove common words when choosing retrieval keywords while retaining them in the original
# response.
STOP_WORDS = set("a an the and or but if of to in on at for from with by as is are was were be been it this that these those i you we they he she my your our their have has had do does did can could would should will what how why which who not".split())


# Normalize a separate text view for search and comparison.
def normalize(text):
    # Apply Unicode NFKC, collapse runs of whitespace and trim the ends.
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


# Make a stable key for text such as a learner's topic name.
def text_hash(text):
    # Hash the normalized, case-folded text so spacing and case variations share a key.
    return hashlib.sha256(normalize(text).casefold().encode()).hexdigest()


# Collect features from the answer while preserving the original wording for quotation.
def analyze(text):
    # Find numbers, words, punctuation and math symbols and record their original character bounds.
    spans = [{"token": m.group(), "start": m.start(), "end": m.end()} for m in re.finditer(r"\d+(?:\.\d+)?|[^\W\d_]+(?:['’][^\W\d_]+)?|[^\w\s]|[\w]", text, re.UNICODE)]
    # Normalize the token strings in a separate view for keyword counting.
    tokens = [unicodedata.normalize("NFKC", item["token"]).casefold() for item in spans]
    # Pick up to 16 frequent terms longer than two characters after excluding common words.
    keywords = [word for word, _ in Counter(t for t in tokens if len(t) > 2 and t not in STOP_WORDS).most_common(16)]
    # Save approximate sentence spans and their offsets in the original text.
    sentences = [{"text": m.group(), "start": m.start(), "end": m.end()} for m in re.finditer(r"[^\n]+?(?:[.!?]+(?=\s|$)|(?=\n)|$)", text) if m.group().strip()]
    # Match common typographic apostrophes in the lexical view only. Source
    # offsets, exact-quote validation and response identity keep the raw text.
    # Standardize apostrophes in the searchable cue text so forms like curly-apostrophe
    # contractions match.
    lexical = normalize(text).casefold().replace('’', "'").replace('‘', "'")
    # Return the raw and normalized versions, plus the original spans and token count.
    return {"raw_text": text, "normalized_text": normalize(text), "token_spans": spans, "token_count": len(tokens),
            # Hash the whitespace-collapsed raw answer. Case and symbols still affect this response
            # fingerprint.
            "sentences": sentences, "keywords": keywords, "response_hash": hashlib.sha256(re.sub(r"\s+", " ", text).strip().encode()).hexdigest(),
            # Record phrases that may reverse the meaning of a learner's statement.
            "negation_cues": re.findall(r"\b(?:not|never|cannot|don't|didn't|isn't|can't|won't|wasn't|aren't|doesn't|couldn't|wouldn't|shouldn't)\b", lexical),
            # Capture uncertainty signals for inspection alongside the explanation.
            "uncertainty_cues": re.findall(r"\b(?:guess|unsure|maybe|not sure|don't know|do not know)\b", lexical),
            # Attach the preprocessing version to the record.
            "version": PREPROCESSING_VERSION}


# Split reference material into overlapping passages for local retrieval.
def chunk_text(text, size=850, overlap=140):
    # Start at the first character of the original document.
    start = 0
    # Continue while there is any source text left to index.
    while start < len(text):
        # Aim for one chunk of the requested size, bounded by the end of the source.
        end = min(start + size, len(text))
        # Try to cut at a word boundary if the passage would stop inside the source.
        if end < len(text):
            # Look for a space in the second half of the current window.
            boundary = text.rfind(" ", start + size // 2, end)
            # Use it only when it yields a valid end after the start.
            if boundary > start:
                # End the chunk at that space.
                end = boundary
        # Yield the unchanged text slice and its original start and end offsets.
        yield {"content": text[start:end], "start": start, "end": end}
        # Stop when the current passage reaches the end.
        if end == len(text):
            # Avoid producing one more overlapping passage at the end of the document.
            break
        # Move forward while intentionally repeating part of the previous chunk.
        start = max(start + 1, end - overlap)


# Turn a concept description and its saved gaps into a SQLite full-text query.
def search_terms(text):
    # Quote up to 14 keywords and join with OR so a passage can match any of the terms.
    return " OR ".join('"' + word.replace('"', '""') + '"' for word in analyze(text)["keywords"][:14])
