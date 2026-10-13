import math
from pathlib import Path

import pytest

from recommender import Recommender, Role, cosine_similarity, normalize

DATA = Path(__file__).resolve().parent.parent / "data" / "raw_skills.csv"


@pytest.fixture(scope="module")
def engine():
    return Recommender.from_csv(DATA)


def test_cosine_identical_and_orthogonal():
    assert math.isclose(cosine_similarity([1, 2, 3], [2, 4, 6]), 1.0)  # magnitude-invariant
    assert cosine_similarity([1, 0], [0, 1]) == 0.0


def test_cosine_zero_vector_is_safe():
    assert cosine_similarity([0, 0], [1, 1]) == 0.0


def test_normalize_aliases():
    assert normalize("  Cloud   Computing ") == "cloud"
    assert normalize("K8s") == "kubernetes"


def test_idf_penalises_common_terms():
    roles = [
        Role("A", ("python", "tensorflow")),
        Role("B", ("python", "docker")),
        Role("C", ("python", "react")),
    ]
    rec = Recommender(roles)
    assert rec.idf["python"] == 0.0
    assert rec.idf["tensorflow"] > rec.idf["python"]


def test_devops_user_gets_devops_first(engine):
    result = engine.recommend(["Docker", "Kubernetes", "CI/CD"])
    assert result.recommendations[0].role == "DevOps Engineer"


def test_assignment_example(engine):
    result = engine.recommend(["Python", "Cloud Computing", "Automation"])
    assert len(result.recommendations) == 3
    assert not result.cold_start
    scores = [r.score for r in result.recommendations]
    assert scores == sorted(scores, reverse=True)
    assert all(0.0 <= s <= 1.0 for s in scores)


def test_requires_three_inputs(engine):
    with pytest.raises(ValueError):
        engine.recommend(["python", "sql"])


def test_top_n_filtering(engine):
    assert len(engine.recommend(["python", "sql", "git"], top_n=5).recommendations) == 5


def test_unknown_skills_reported(engine):
    result = engine.recommend(["python", "sql", "underwater basket weaving"])
    assert "underwater basket weaving" in result.unknown_skills


def test_cold_start_falls_back_to_trending(engine):
    result = engine.recommend(["foo", "bar", "baz"])
    assert result.cold_start
    assert result.recommendations[0].role == "Software Engineer"  # highest popularity


def test_deterministic(engine):
    a = engine.recommend(["python", "sql", "git"])
    b = engine.recommend(["git", "sql", "python"])
    assert [r.role for r in a.recommendations] == [r.role for r in b.recommendations]
