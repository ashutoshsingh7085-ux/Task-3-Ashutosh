# 🎯 Tech Stack Recommender

> **DecodeLabs · Artificial Intelligence · Project 3: AI Recommendation Logic**
> A content-based recommendation engine that maps your skills to the best-fit tech career paths, built with **pure Python** (standard library only): TF-IDF weighting + cosine similarity.

```
$ python -m recommender --skills "Python,Cloud Computing,Automation"

Top 3 career paths for: python, cloud, automation

1. Site Reliability Engineer  -  46.4% match
     You have : automation, python, cloud
     Learn next: linux, kubernetes, monitoring, ci/cd, bash

2. Cloud Architect  -  39.5% match
     You have : cloud, automation
     Learn next: aws, azure, networking, security, terraform, kubernetes

3. Cloud Engineer  -  27.2% match
     You have : cloud, python
     Learn next: aws, azure, linux, docker, terraform, networking
```

## ✨ Features

- **Content-based filtering**: no historical user data required.
- **TF-IDF weighting**: rare, specific skills (e.g. `terraform`) count more than generic ones (e.g. `python`).
- **Cosine similarity**: compares the *direction* of preferences, not their magnitude.
- **Top-N output** with a match percentage, the skills you already have, and **what to learn next**.
- **Cold-start handling**: if none of your skills are recognised, falls back to trending roles.
- **Smart input handling**: aliases (`k8s` → `kubernetes`, `Cloud Computing` → `cloud`) and "did you mean…?" hints.
- **Zero dependencies** at runtime. Tests use `pytest`.

## 🏗️ Architecture (Input → Process → Output)

| Step | Stage | What happens |
|------|-------|--------------|
| 1 | **Ingestion** | Capture ≥ 3 skills; normalise them into the dataset's shared vocabulary |
| 2 | **Scoring** | Build TF-IDF vectors, compute cosine similarity against every role |
| 3 | **Sorting** | Rank roles by score, descending (deterministic tie-break) |
| 4 | **Filtering** | Truncate to Top-N (default 3) to prevent choice overload |

### The math

```
TF(t, d)  = count of t in d / total terms in d
IDF(t)    = log( total documents / documents containing t )
weight    = TF × IDF

cos(θ)    = (A · B) / (‖A‖ ‖B‖)      →  range [0, 1] since TF-IDF is non-negative
```

Item (role) vectors and user vectors live in the **same vocabulary space**, which is why input normalisation matters: `"Web Design"` vs `"Frontend Development"` would otherwise break the similarity math.

## 🚀 Quick start

```bash
git clone https://github.com/<your-username>/tech-stack-recommender.git
cd tech-stack-recommender

# Interactive mode
python -m recommender

# One-shot mode
python -m recommender --skills "Java,Algorithms,SQL"

# More results
python -m recommender --skills "python,sql,git" --top 5

# See every skill the engine understands
python -m recommender --list-skills
```

Requires **Python 3.9+**.

### Use it as a library

```python
from recommender import Recommender

engine = Recommender.from_csv("data/raw_skills.csv")
result = engine.recommend(["Python", "Cloud Computing", "Automation"], top_n=3)

for rec in result.recommendations:
    print(rec.role, f"{rec.percent}%", rec.missing)
```

## 🧪 Tests

```bash
pip install -r requirements.txt
pytest
```

Covers cosine math, IDF behaviour, input normalisation, the 3-input minimum, Top-N filtering, cold start, and determinism.

## 📁 Project structure

```
tech-stack-recommender/
├── data/raw_skills.csv        # Job roles (the "items") + skills + popularity
├── recommender/
│   ├── engine.py              # TF-IDF, cosine similarity, ranking pipeline
│   ├── cli.py                 # Interactive + argparse interface
│   └── __main__.py            # `python -m recommender`
├── tests/test_engine.py
└── .github/workflows/ci.yml   # Runs tests on every push
```

## 🧊 Cold-start strategy

| Problem | Mitigation in this project |
|---------|----------------------------|
| **New user** (no usable skills) | Forced onboarding input (≥ 3 skills) + trending-roles fallback by `popularity` |
| **New item** (new role) | Add a row to `raw_skills.csv`; content-based filtering recommends it immediately |

## 🔧 Extending the dataset

Add a row to `data/raw_skills.csv`:

```csv
role,popularity,skills
Blockchain Developer,65,solidity;cryptography;javascript;git;apis;data structures
```

Skills are `;`-separated. Add synonyms to `ALIASES` in `recommender/engine.py`.

## 💡 Ideas to go further

- Let users **rate** skill proficiency (1–5) to weight the user vector
- Add role-to-role similarity ("roles like Data Scientist")
- Swap the CSV for a larger dataset or scrape job postings
- Compare against Jaccard / Euclidean and benchmark the rankings

## 📄 License

MIT. See [LICENSE](LICENSE).

---
*Built as part of the DecodeLabs Industrial Training Kit (Batch 2026).*
