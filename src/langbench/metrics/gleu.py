"""GLEU for grammatical error correction (Napoles et al., 2015; 2016 update).

Sentence-level GLEU against one reference:
  For n = 1..4, precision_n =
      ( matches(cand, ref) - penalty(cand, src, ref) ) / |cand n-grams|
  where penalty counts candidate n-grams that appear in the SOURCE but not in
  the REFERENCE (the model kept an error the reference fixed, or introduced
  source-like text). Clip precisions at 0, take the geometric mean over n,
  apply the BLEU brevity penalty against the reference length.

Multiple references: mean of per-reference GLEU. The Napoles 2016 update
samples one reference per bootstrap iteration; with the small per-item
reference counts here (1-2), the mean is equivalent in expectation.
# DECISION: mean over references.
# VERIFIED 2026-07-25 against the reference implementation
# (github.com/cnap/gec-ranking, sentence-level smooth=True path): 10/10 real
# W&I items match exactly after adopting the reference's two behaviors
# (DECISION 36): (a) sentence-level smoothing replaces zero numerator/
# denominator stats with 1 (not a log floor); (b) the source-ngram penalty
# uses a SET difference (an n-gram type present in the reference is never
# penalized, regardless of source count surplus). Empty candidates still
# score 0.0 (guard; the pipeline maps parse failures to 0 before this).

Tokenization: whitespace on already-tokenized data (W&I M2 is tokenized);
simple punctuation-splitting fallback for raw text (MERLIN, COWS-L2H).
Corpus-level ranking is done on per-item GLEU means with bootstrap CIs
(bootstrap.py), so per-item scores are what this module returns.
"""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_gleu(source: str, candidate: str, references: list[str], max_n: int = 4) -> float:
    if not references:
        raise ValueError("sentence_gleu requires at least one reference")
    scores = [
        _gleu_single(tokenize(source), tokenize(candidate), tokenize(ref), max_n)
        for ref in references
    ]
    return sum(scores) / len(scores)


def _gleu_single(
    src: list[str], cand: list[str], ref: list[str], max_n: int
) -> float:
    if not cand:
        return 0.0
    log_sum = 0.0
    for n in range(1, max_n + 1):
        cand_n = _ngrams(cand, n)
        ref_n = _ngrams(ref, n)
        src_n = _ngrams(src, n)
        matches = sum((cand_n & ref_n).values())
        # Reference behavior (set difference): an n-gram TYPE that appears in
        # the reference is never penalized, even if the source has surplus
        # count. (Counter subtraction would penalize the surplus.)
        src_only = Counter({k: v for k, v in src_n.items() if k not in ref_n})
        penalty = sum((cand_n & src_only).values())
        num = max(matches - penalty, 0)
        den = sum(cand_n.values())
        # Reference sentence-level smoothing (gleu(stats, smooth=True)):
        # any zero stat — numerator or denominator — becomes 1.
        log_sum += math.log((num or 1) / (den or 1))
    geo_mean = math.exp(log_sum / max_n)
    bp = 1.0 if len(cand) >= len(ref) else math.exp(1 - len(ref) / max(len(cand), 1))
    return bp * geo_mean


def edit_rate(source: str, candidate: str) -> float:
    """Fraction of source tokens changed (token-level Levenshtein / |src|).

    Sanity signal: models that rewrite everything get high GLEU-irrelevant
    churn; the report flags items with edit_rate above REWRITE_THRESHOLD.
    """
    s, c = tokenize(source), tokenize(candidate)
    if not s:
        return 0.0 if not c else 1.0
    prev = list(range(len(c) + 1))
    for i in range(1, len(s) + 1):
        cur = [i] + [0] * len(c)
        for j in range(1, len(c) + 1):
            cost = 0 if s[i - 1] == c[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[len(c)] / len(s)


REWRITE_THRESHOLD = 0.5  # DECISION: >50% of tokens changed => flagged as rewrite
