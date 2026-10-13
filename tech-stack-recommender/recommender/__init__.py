"""Tech Stack Recommender: content-based filtering with TF-IDF + cosine similarity."""

from .engine import (
    Recommendation,
    RecommendationResult,
    Recommender,
    Role,
    cosine_similarity,
    normalize,
)

__all__ = [
    "Recommendation",
    "RecommendationResult",
    "Recommender",
    "Role",
    "cosine_similarity",
    "normalize",
]
