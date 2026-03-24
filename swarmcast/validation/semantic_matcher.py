"""Semantic similarity matching for backtest validation."""

from __future__ import annotations

from typing import Callable

import numpy as np


class SemanticMatcher:
    """Match simulation outputs to expected outcomes using embeddings."""

    def __init__(
        self,
        embedder: Callable[[str], np.ndarray],
        similarity_threshold: float = 0.45,
    ):
        """Initialize with a callable embedder and a cosine similarity threshold."""
        self._embedder = embedder
        self._threshold = similarity_threshold
        self._cache: dict[str, np.ndarray] = {}

    def _embed(self, text: str) -> np.ndarray:
        """Return the embedding for text, computing and caching it on first call."""
        if text not in self._cache:
            self._cache[text] = self._embedder(text)
        return self._cache[text]

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors, returning 0.0 for zero-norm inputs."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def match_score(self, expected: str, actual: str) -> float:
        """Return the cosine similarity between the embeddings of expected and actual."""
        emb_expected = self._embed(expected)
        emb_actual = self._embed(actual)
        return self.cosine_similarity(emb_expected, emb_actual)

    def best_match(
        self,
        expected: str,
        candidates: list[str],
    ) -> tuple[float, str, int]:
        """Find the best matching candidate for an expected outcome."""
        if not candidates:
            return 0.0, "", -1

        emb_expected = self._embed(expected)
        best_score = 0.0
        best_text = ""
        best_idx = -1

        for i, candidate in enumerate(candidates):
            emb_candidate = self._embed(candidate)
            score = self.cosine_similarity(emb_expected, emb_candidate)
            if score > best_score:
                best_score = score
                best_text = candidate
                best_idx = i

        return best_score, best_text, best_idx

    def is_match(self, expected: str, actual: str) -> bool:
        """Return True if the match score meets or exceeds the similarity threshold."""
        return self.match_score(expected, actual) >= self._threshold

    def find_matches(
        self,
        expected: str,
        candidates: list[str],
        top_k: int = 5,
    ) -> list[tuple[float, str, int]]:
        """Return the top-k candidates ranked by cosine similarity to expected."""
        emb_expected = self._embed(expected)
        scored = []

        for i, candidate in enumerate(candidates):
            emb_candidate = self._embed(candidate)
            score = self.cosine_similarity(emb_expected, emb_candidate)
            scored.append((score, candidate, i))

        scored.sort(key=lambda x: -x[0])
        return scored[:top_k]

    def bulk_match(
        self,
        expected_outcomes: list[str],
        simulation_texts: list[str],
    ) -> list[dict]:
        """Match each expected outcome against all simulation texts and return scored result dicts."""
        results = []

        for expected in expected_outcomes:
            top_matches = self.find_matches(expected, simulation_texts, top_k=3)
            best_score = top_matches[0][0] if top_matches else 0.0
            predicted = best_score >= self._threshold

            results.append({
                "expected": expected,
                "predicted": predicted,
                "best_score": best_score,
                "threshold": self._threshold,
                "top_matches": [
                    {"score": score, "text": text[:200], "index": idx}
                    for score, text, idx in top_matches
                ],
            })

        return results
