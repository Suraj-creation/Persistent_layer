from __future__ import annotations

import math
from typing import Dict, List, Tuple
from ..evidence.models import ConceptLink
from .taxonomy import FRACTION_TAXONOMY, KnowledgeComponent


class SemanticTaxonomyMatcher:
    """
    Matches extracted claims to Knowledge Components in the curriculum taxonomy.
    Supports SentenceTransformers dense embeddings with an exact lexical fallback.
    """

    def __init__(self, taxonomy: Optional[Dict[str, KnowledgeComponent]] = None, use_embeddings: bool = True):
        self.taxonomy = taxonomy or FRACTION_TAXONOMY
        self.use_embeddings = use_embeddings
        self._model = None

    def _get_embedding_model(self):
        if self._model is None and self.use_embeddings:
            try:
                from sentence_transformers import SentenceTransformer
                # Use lightweight, fast SentenceTransformer
                self._model = SentenceTransformer("all-MiniLM-L6-v2")
            except Exception:
                self._model = False
        return self._model if self._model is not False else None

    def _lexical_similarity(self, text1: str, text2: str) -> float:
        """Simple fast token overlap similarity as fallback."""
        words1 = set(text1.lower().replace("_", " ").split())
        words2 = set(text2.lower().replace("_", " ").split())
        if not words1 or not words2:
            return 0.0
        intersection = words1.intersection(words2)
        return len(intersection) / math.sqrt(len(words1) * len(words2))

    def match(self, claim_text: str, top_k: int = 2) -> List[ConceptLink]:
        """
        Finds the top-k best matching Knowledge Components for a given claim.
        Returns a list of ConceptLink objects with confidence scores.
        """
        model = self._get_embedding_model()
        scored_concepts: List[Tuple[KnowledgeComponent, float, str]] = []

        if model is not None:
            try:
                import numpy as np
                claim_emb = model.encode(claim_text, convert_to_numpy=True)
                for kc in self.taxonomy.values():
                    kc_text = f"{kc.concept_name}. {kc.description}"
                    kc_emb = model.encode(kc_text, convert_to_numpy=True)
                    # Cosine similarity
                    sim = float(np.dot(claim_emb, kc_emb) / (np.linalg.norm(claim_emb) * np.linalg.norm(kc_emb)))
                    # Normalize to 0..1
                    norm_sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))
                    rationale = f"Dense semantic match score {norm_sim:.2f} with '{kc.concept_name}'."
                    scored_concepts.append((kc, norm_sim, rationale))
            except Exception:
                model = None  # Fallback to lexical

        if model is None:
            # Lexical fallback
            for kc in self.taxonomy.values():
                target_text = f"{kc.concept_name} {kc.description} {kc.concept_id}"
                sim = self._lexical_similarity(claim_text, target_text)
                rationale = f"Lexical keyword overlap {sim:.2f} with '{kc.concept_name}'."
                scored_concepts.append((kc, min(1.0, sim * 1.5), rationale))

        # Sort descending by score
        scored_concepts.sort(key=lambda x: x[1], reverse=True)

        results = []
        for kc, score, rationale in scored_concepts[:top_k]:
            results.append(
                ConceptLink(
                    concept_id=kc.concept_id,
                    concept_name=kc.concept_name,
                    confidence=round(score, 3),
                    grounding_rationale=rationale,
                )
            )

        return results
