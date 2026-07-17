"""
MetricGuard - Semantic Matching Engine
========================================

WHAT THIS FILE TEACHES YOU (read these, they ARE your interview answers):

1. EMBEDDINGS: turning text into a list of numbers (a "vector") that captures
   MEANING. Two texts with similar meaning get vectors that point in a similar
   direction, even if they use different words ("MAU" vs "monthly active users").

2. VECTOR DB / SIMILARITY SEARCH: once everything is a vector, we can measure
   how "close" two pieces of text are using COSINE SIMILARITY. That's literally
   what a Vector DB (Pinecone, FAISS, Chroma) does under the hood at scale.

3. THE CORE IDEA OF THIS PROJECT: metrics with the SAME meaning but DIFFERENT
   definitions are the silent killer of trust in company dashboards. We detect
   them automatically.
"""

import json
import numpy as np
from pathlib import Path

DATA = Path(__file__).parent.parent / "data" / "metric_definitions.json"


def load_metrics():
    """Load the raw metric definitions from our synthetic 'company'."""
    with open(DATA) as f:
        return json.load(f)


# A tiny "concept glossary": maps different words teams use to a shared concept.
# WHY THIS MATTERS: this is a miniature version of the SEMANTIC GLOSSARY that a
# real RAG system retrieves from. It's how you ground different vocabulary
# ('MAU', 'active users', 'logins') into one shared meaning. With strong neural
# embeddings you need this less; with TF-IDF it's essential. Either way, knowing
# WHY you'd keep a governed glossary is a strong interview point.
CONCEPT_GLOSSARY = {
    "mau": "monthly active users concept",
    "monthly active users": "monthly active users concept",
    "active users": "monthly active users concept",
    "monthly_active_users": "monthly active users concept",
    "active_users": "monthly active users concept",
    "revenue": "revenue concept",
    "monthly_revenue": "revenue concept",
    "revenue_monthly": "revenue concept",
    "churn": "churn concept",
    "churn_rate": "churn concept",
    "customer_churn": "churn concept",
    "conversion_rate": "conversion concept",
    "average_order_value": "average order value concept",
    "aov": "average order value concept",
}


def concept_tags(metric):
    """Look up which shared concept(s) this metric touches, from its name."""
    name = metric["metric_name"].lower()
    tags = []
    for key, concept in CONCEPT_GLOSSARY.items():
        if key in name or key.replace("_", " ") in name.replace("_", " "):
            tags.append(concept)
    return " ".join(sorted(set(tags)))


def build_text(metric):
    """
    Combine the fields of a metric into one text blob to embed.
    WHY: the *meaning* of a metric lives in its name + description + logic
    together, not any single field. We also append the shared concept tag so
    that 'MAU' and 'active_users' land near each other in vector space.
    """
    # We repeat the concept tag a few times so it strongly influences the vector.
    # (With neural embeddings this weighting matters less, but it's a clean,
    # explainable way to make the shared concept the dominant signal.)
    concept = (concept_tags(metric) + " ") * 5
    return (
        f"{concept}"
        f"Metric name: {metric['metric_name']}. "
        f"Description: {metric['description']}. "
        f"SQL logic: {metric['sql']}"
    )


# -----------------------------------------------------------------------------
# EMBEDDINGS
# -----------------------------------------------------------------------------
# We use a small, free, LOCAL embedding model (sentence-transformers) so this
# runs on your machine with NO API key. In a real company you'd call an
# embeddings API (OpenAI, Cohere, Anthropic-adjacent) - the CONCEPT is identical.

_model = None

def embed(texts):
    """
    Turn a list of texts into a matrix of vectors (one vector per text).

    We try TWO ways, best first:
      (A) A real neural embedding model (sentence-transformers). Best quality:
          understands that 'MAU' and 'monthly active users' mean the same thing.
      (B) FALLBACK: TF-IDF (classic NLP). Turns text into word-frequency vectors.
          Runs 100% offline with no downloads. Lower quality on synonyms, but the
          CONCEPT you explain in an interview is identical: text -> vector ->
          cosine similarity. In production you'd always use (A).

    Either way we L2-normalize vectors to length 1 so cosine similarity == dot
    product (fast) and the rest of the pipeline doesn't care which was used.
    """
    global _model
    # --- (A) try the neural model ---
    try:
        if _model is None:
            from sentence_transformers import SentenceTransformer
            _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model.encode(texts, normalize_embeddings=True)
    except Exception:
        pass  # no internet / model unavailable -> fall through to TF-IDF

    # --- (B) offline fallback: TF-IDF ---
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize
    vec = TfidfVectorizer(stop_words="english")
    matrix = vec.fit_transform(texts).toarray()
    return normalize(matrix)  # L2-normalize each row to length 1


# -----------------------------------------------------------------------------
# COSINE SIMILARITY  (the "search" in Vector DB / similarity search)
# -----------------------------------------------------------------------------
def cosine_similarity_matrix(vectors):
    """
    Compare EVERY metric against EVERY other metric.

    Cosine similarity measures the ANGLE between two vectors:
      - 1.0  => same direction => same meaning
      - 0.0  => unrelated
    Because we normalized vectors to length 1, the whole all-pairs comparison
    is just one matrix multiply: V @ V.T. This is exactly what a Vector DB does,
    just optimized for millions of vectors.
    """
    return vectors @ vectors.T


# -----------------------------------------------------------------------------
# CONFLICT DETECTION
# -----------------------------------------------------------------------------
def find_semantic_groups(metrics, sim_matrix, threshold=0.72):
    """
    Group metrics that MEAN the same thing (high similarity) even if named
    differently. threshold=0.72 means "72% similar in meaning or more".

    This is where 'active_users', 'MAU', and 'monthly_active_users' get pulled
    into ONE group despite different names -> that's the semantic magic.
    """
    n = len(metrics)
    visited = set()
    groups = []
    for i in range(n):
        if i in visited:
            continue
        group = [i]
        visited.add(i)
        for j in range(i + 1, n):
            if j not in visited and sim_matrix[i][j] >= threshold:
                group.append(j)
                visited.add(j)
        groups.append(group)
    return [g for g in groups if len(g) > 1]  # only groups with a potential conflict


def detect_definition_conflicts(metrics, group):
    """
    Given a group of metrics that MEAN the same thing, check whether their
    actual LOGIC differs. Same meaning + different logic = a TRUST CONFLICT.

    We compare concrete, checkable attributes:
      - refund handling
      - time grain (rolling 30d vs calendar month vs all-time)
      - filters applied
    These are the real-world causes of 'same metric, different number'.
    """
    conflicts = []
    members = [metrics[i] for i in group]

    # refund handling mismatch
    refund_flags = {m.get("includes_refunds") for m in members if "includes_refunds" in m}
    if len(refund_flags) > 1:
        conflicts.append("Refund handling differs (some include refunds, some don't)")

    # time grain mismatch
    grains = {m.get("time_grain") for m in members}
    if len(grains) > 1:
        conflicts.append(f"Time window differs: {sorted(grains)}")

    # filter mismatch
    filtersets = {tuple(sorted(m.get("filters", []))) for m in members}
    if len(filtersets) > 1:
        conflicts.append("Filter logic differs across definitions")

    return conflicts


def trust_risk_score(group, conflicts):
    """
    Turn conflicts into a 0-100 'trust risk' number an executive can read.
    More teams involved + more conflicts = higher risk that dashboards disagree.
    THIS is the 'so what' that makes the output business-usable.
    """
    teams_involved = len(group)
    base = min(100, teams_involved * 15 + len(conflicts) * 20)
    return base


def run_analysis(threshold=0.72):
    """Full pipeline: load -> embed -> similarity -> group -> detect -> score."""
    metrics = load_metrics()
    texts = [build_text(m) for m in metrics]
    vectors = embed(texts)
    sim = cosine_similarity_matrix(vectors)
    groups = find_semantic_groups(metrics, sim, threshold)

    results = []
    for group in groups:
        conflicts = detect_definition_conflicts(metrics, group)
        if not conflicts:
            continue  # same meaning AND same logic = fine, not a conflict
        results.append({
            "metrics": [metrics[i] for i in group],
            "metric_ids": [metrics[i]["id"] for i in group],
            "teams": sorted({metrics[i]["team"] for i in group}),
            "names": sorted({metrics[i]["metric_name"] for i in group}),
            "conflicts": conflicts,
            "trust_risk": trust_risk_score(group, conflicts),
            "avg_similarity": float(np.mean([sim[i][j] for i in group for j in group if i < j])),
        })
    results.sort(key=lambda r: r["trust_risk"], reverse=True)
    return metrics, results


if __name__ == "__main__":
    metrics, results = run_analysis()
    print(f"\nAnalyzed {len(metrics)} metric definitions.")
    print(f"Found {len(results)} conflicting metric groups.\n")
    for r in results:
        print("=" * 70)
        print(f"CONFLICT  |  trust risk: {r['trust_risk']}/100  |  similarity: {r['avg_similarity']:.2f}")
        print(f"  Names used : {', '.join(r['names'])}")
        print(f"  Teams      : {', '.join(r['teams'])}")
        print(f"  Metric IDs : {', '.join(r['metric_ids'])}")
        print(f"  Problems   :")
        for c in r["conflicts"]:
            print(f"     - {c}")
    print("=" * 70)
