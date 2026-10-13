"""Core recommendation engine (pure Python, standard library only).

Pipeline (Input -> Process -> Output):
    1. Ingestion : user skills are normalised into the dataset's shared vocabulary.
    2. Scoring   : TF-IDF vectors are built; cosine similarity is computed per role.
    3. Sorting   : roles are ranked by score (descending).
    4. Filtering : the list is truncated to the Top-N.
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

# Maps common spellings/synonyms onto the dataset's canonical vocabulary.
# Item and user features MUST share one vocabulary, otherwise the math fails.
ALIASES: dict[str, str] = {
    "cloud computing": "cloud",
    "amazon web services": "aws",
    "k8s": "kubernetes",
    "cicd": "ci/cd",
    "ci cd": "ci/cd",
    "continuous integration": "ci/cd",
    "ml": "machine learning",
    "dl": "deep learning",
    "natural language processing": "nlp",
    "js": "javascript",
    "node": "node.js",
    "nodejs": "node.js",
    "ts": "typescript",
    "scripting languages": "scripting",
    "api": "apis",
    "rest": "apis",
    "rest api": "apis",
    "dsa": "data structures",
    "tf": "tensorflow",
    "powerbi": "power bi",
    "ms excel": "excel",
    "pentesting": "penetration testing",
    "ethical hacking": "penetration testing",
    "ui/ux": "ui design",
    "ux": "ui design",
    "shell": "bash",
    "shell scripting": "bash",
    "data viz": "data visualization",
    "visualization": "data visualization",
}


def normalize(skill: str) -> str:
    """Lower-case, collapse whitespace and resolve aliases."""
    cleaned = " ".join(skill.strip().lower().split())
    return ALIASES.get(cleaned, cleaned)


@dataclass(frozen=True)
class Role:
    name: str
    skills: tuple[str, ...]
    popularity: int = 0


@dataclass(frozen=True)
class Recommendation:
    role: str
    score: float  # cosine similarity in [0, 1]
    matched: tuple[str, ...]  # skills the user already has for this role
    missing: tuple[str, ...]  # skills the user should learn next

    @property
    def percent(self) -> float:
        return round(self.score * 100, 1)


@dataclass(frozen=True)
class RecommendationResult:
    recommendations: list[Recommendation]
    known_skills: list[str]
    unknown_skills: list[str] = field(default_factory=list)
    cold_start: bool = False  # True when we fell back to trending roles


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """cos(theta) = (A . B) / (||A|| * ||B||); returns 0.0 for zero vectors."""
    if len(a) != len(b):
        raise ValueError("Vectors must have the same dimensionality")
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class Recommender:
    """Content-based recommender that treats job roles as the 'items'."""

    def __init__(self, roles: list[Role]):
        if not roles:
            raise ValueError("At least one role is required")
        self.roles = roles
        self.vocabulary: list[str] = sorted({s for r in roles for s in r.skills})
        self._index = {skill: i for i, skill in enumerate(self.vocabulary)}
        self.idf = self._compute_idf()
        self._role_vectors = [self._vectorize(list(r.skills)) for r in roles]

    # ---------- construction ----------
    @classmethod
    def from_csv(cls, path: str | Path) -> "Recommender":
        roles: list[Role] = []
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                skills = tuple(
                    dict.fromkeys(  # dedupe, keep order
                        normalize(s) for s in row["skills"].split(";") if s.strip()
                    )
                )
                roles.append(
                    Role(
                        name=row["role"].strip(),
                        skills=skills,
                        popularity=int(row.get("popularity") or 0),
                    )
                )
        return cls(roles)

    # ---------- TF-IDF ----------
    def _compute_idf(self) -> dict[str, float]:
        """IDF = log(total documents / documents containing term)."""
        n_docs = len(self.roles)
        doc_freq = Counter(s for r in self.roles for s in set(r.skills))
        return {term: math.log(n_docs / df) for term, df in doc_freq.items()}

    def _vectorize(self, terms: list[str]) -> list[float]:
        """TF = count(t in d) / total terms in d; weight = TF * IDF."""
        in_vocab = [t for t in terms if t in self._index]
        vector = [0.0] * len(self.vocabulary)
        if not in_vocab:
            return vector
        total = len(in_vocab)
        for term, count in Counter(in_vocab).items():
            vector[self._index[term]] = (count / total) * self.idf[term]
        return vector

    # ---------- ingestion ----------
    def parse_skills(self, raw: list[str]) -> tuple[list[str], list[str]]:
        """Split raw input into (known, unknown) skills, de-duplicated."""
        known: list[str] = []
        unknown: list[str] = []
        for item in raw:
            skill = normalize(item)
            if not skill:
                continue
            bucket = known if skill in self._index else unknown
            if skill not in bucket:
                bucket.append(skill)
        return known, unknown

    # ---------- recommend ----------
    def recommend(
        self, user_skills: list[str], top_n: int = 3, min_inputs: int = 3
    ) -> RecommendationResult:
        known, unknown = self.parse_skills(user_skills)
        if len(known) + len(unknown) < min_inputs:
            raise ValueError(f"Please provide at least {min_inputs} distinct skills.")

        user_vector = self._vectorize(known)

        # Cold start: nothing we can embed -> fall back to trending roles.
        if not any(user_vector):
            trending = sorted(self.roles, key=lambda r: (-r.popularity, r.name))
            recs = [Recommendation(r.name, 0.0, (), r.skills) for r in trending[:top_n]]
            return RecommendationResult(recs, known, unknown, cold_start=True)

        # Step 2: scoring
        scored = [
            (cosine_similarity(user_vector, vec), role)
            for vec, role in zip(self._role_vectors, self.roles)
        ]
        # Step 3: sorting (deterministic tie-break on name)
        scored.sort(key=lambda pair: (-pair[0], pair[1].name))
        # Step 4: filtering (Top-N)
        recs = []
        for score, role in scored[:top_n]:
            matched = tuple(s for s in role.skills if s in known)
            missing = tuple(s for s in role.skills if s not in known)
            recs.append(Recommendation(role.name, score, matched, missing))
        return RecommendationResult(recs, known, unknown)
