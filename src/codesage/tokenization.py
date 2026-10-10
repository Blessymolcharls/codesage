"""Code-aware tokenization and text preprocessing for lexical code search."""

import re
from typing import List


# Regex pattern to match camelCase, PascalCase, snake_case, and other identifier boundaries
_CAMEL_CASE_PATTERN = re.compile(r"([a-z0-9])([A-Z])")
_PUNCT_SPLIT_PATTERN = re.compile(r"[^a-zA-Z0-9_]+")
_NON_ALPHANUM_PATTERN = re.compile(r"[^a-zA-Z0-9]+")


def split_identifier(token: str) -> List[str]:
    """Split programming identifier into constituent subwords while keeping the original.
    
    Example:
        'validateUploadedPdf' -> ['validateuploadedpdf', 'validate', 'uploaded', 'pdf']
        'pdf_merge_service'   -> ['pdf_merge_service', 'pdf', 'merge', 'service']
    """
    token_lower = token.lower()
    results = [token_lower]

    # Split CamelCase/PascalCase
    subwords_camel = _CAMEL_CASE_PATTERN.sub(r"\1 \2", token).lower()

    # Split on underscores, hyphens, and any non-alphanumeric chars
    subwords = _NON_ALPHANUM_PATTERN.split(subwords_camel)

    for sw in subwords:
        sw = sw.strip()
        if len(sw) > 1 and sw not in results:
            results.append(sw)

    return results


def code_aware_tokenize(text: str) -> List[str]:
    """Tokenize source code or queries, preserving both compound identifiers and decomposed tokens."""
    if not text:
        return []

    tokens: List[str] = []
    # Split initial text by whitespace and common punctuation
    raw_tokens = _PUNCT_SPLIT_PATTERN.split(text)

    for raw in raw_tokens:
        raw = raw.strip()
        if not raw or len(raw) < 2:
            continue

        # Check if it has snake_case or CamelCase
        if "_" in raw or any(c.isupper() for c in raw[1:]):
            sub_tokens = split_identifier(raw)
            tokens.extend(sub_tokens)
        else:
            tokens.append(raw.lower())

    return tokens


def prepare_search_document(
    symbol_name: str | None,
    rel_path: str,
    source_code: str,
    language: str | None = None,
    unit_type: str | None = None,
    include_metadata: bool = True,
) -> str:
    """Compose a structured search document representation with weighted metadata fields."""
    if not include_metadata:
        return source_code

    parts: List[str] = []

    # Symbol name has highest salience: boost by repeating 3 times
    if symbol_name:
        parts.extend([symbol_name] * 3)

    # Relative path has strong domain signal: boost by repeating 2 times
    path_cleaned = rel_path.replace("/", " ").replace("\\", " ").replace(".", " ")
    parts.extend([path_cleaned] * 2)

    # Language and unit type
    if language:
        parts.append(language)
    if unit_type:
        parts.append(unit_type)

    # Source code content
    parts.append(source_code)

    return " \n ".join(parts)
