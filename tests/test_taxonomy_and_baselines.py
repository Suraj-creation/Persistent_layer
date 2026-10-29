from __future__ import annotations

import pandas as pd
import pytest
from packages.learner_state_core.grounding.matcher import SemanticTaxonomyMatcher
from packages.learner_state_core.grounding.taxonomy import FRACTION_TAXONOMY
from packages.learner_state_core.baselines.bkt import ClassicalBKTBaseline


def test_semantic_taxonomy_matcher():
    """Verify that semantic taxonomy matcher correctly links fraction claims to curriculum KCs."""
    matcher = SemanticTaxonomyMatcher(taxonomy=FRACTION_TAXONOMY, use_embeddings=False)
    
    # Claim concerning adding unlike denominators
    links = matcher.match("I added fractions with different denominators", top_k=2)
    assert len(links) >= 1
    top_link = links[0]
    assert "unlike" in top_link.concept_id or "fraction" in top_link.concept_id
    assert top_link.confidence > 0.0


def test_classical_bkt_baseline():
    """Verify that pyBKT baseline can fit sequential synthetic data and evaluate metrics."""
    # Synthetic sequence data matching pyBKT requirements
    data = {
        "user_id": [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
        "skill_name": ["fraction_addition"] * 16,
        "correct": [0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1],
        "order_id": list(range(1, 17)),
    }
    df = pd.DataFrame(data)

    bkt = ClassicalBKTBaseline(seed=42)
    bkt.fit(df, skill_name="fraction_addition")

    eval_results = bkt.evaluate(df, skill_name="fraction_addition")
    assert "auc" in eval_results
    assert "accuracy" in eval_results
    assert "brier_score" in eval_results
    assert "ece" in eval_results
    assert 0.0 <= eval_results["accuracy"] <= 1.0
    assert 0.0 <= eval_results["ece"] <= 1.0
