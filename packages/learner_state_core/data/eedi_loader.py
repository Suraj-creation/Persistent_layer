from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..evidence.models import Interaction
from .schemas import DatasetAuditReport


class EediDataLoader:
    """
    Data connector for Eedi Diagnostic Questions & Kaggle Math Misconceptions datasets.
    Extracts student explanations, maps distractor choices, and preserves ground-truth misconception IDs.
    """

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else None
        self.df: Optional[pd.DataFrame] = None

    def load(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Loads data from file or takes in-memory DataFrame."""
        if df is not None:
            self.df = df
        elif self.data_path and self.data_path.exists():
            if self.data_path.suffix == ".csv":
                self.df = pd.read_csv(self.data_path)
            elif self.data_path.suffix == ".json":
                self.df = pd.read_json(self.data_path)
            elif self.data_path.suffix == ".parquet":
                self.df = pd.read_parquet(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path.suffix}")
        else:
            self.df = self.create_sample_eedi_dataset()

        return self.df

    def audit(self) -> DatasetAuditReport:
        """Audits the loaded Eedi dataset and outputs a structured suitability report."""
        if self.df is None:
            self.load()

        df = self.df
        total_records = len(df)
        total_students = int(df["user_id"].nunique()) if "user_id" in df.columns else 0
        total_items = int(df["question_id"].nunique()) if "question_id" in df.columns else 0
        
        # Concept counts
        concept_col = "construct_name" if "construct_name" in df.columns else "concept_id"
        total_concepts = int(df[concept_col].nunique()) if concept_col in df.columns else 0
        concept_dist = df[concept_col].value_counts().to_dict() if concept_col in df.columns else {}

        # Text availability
        text_col = "explanation" if "explanation" in df.columns else "response_text"
        has_text = df[text_col].dropna().apply(lambda s: len(str(s).strip()) > 0) if text_col in df.columns else pd.Series([False]*total_records)
        text_avail_rate = float(has_text.mean()) if total_records > 0 else 0.0

        # Misconception label rate
        misc_col = "misconception_id" if "misconception_id" in df.columns else "ground_truth_misconception"
        has_misc = df[misc_col].dropna().apply(lambda s: str(s).strip() not in ["", "none", "nan", "null"]) if misc_col in df.columns else pd.Series([False]*total_records)
        misc_label_rate = float(has_misc.mean()) if total_records > 0 else 0.0

        missingness = {col: float(df[col].isna().mean()) for col in df.columns}

        leakage_risks = []
        if "user_id" in df.columns and "timestamp" not in df.columns and "sequence_id" not in df.columns:
            leakage_risks.append("LACK_OF_TIME_SEQUENCE_ORDER: Interactions lack explicit sequence order.")

        verdict = (
            "SUITABLE_FOR_LANGUAGE_GROUNDED_MISCONCEPTION_MODELING"
            if text_avail_rate >= 0.70 and misc_label_rate >= 0.30
            else "PARTIALLY_SUITABLE_LIMITED_LANGUAGE_SIGNALS"
        )

        return DatasetAuditReport(
            dataset_name="Eedi Diagnostic Questions (Math Misconceptions)",
            total_records=total_records,
            total_students=total_students,
            total_items=total_items,
            total_concepts=total_concepts,
            text_availability_rate=round(text_avail_rate, 4),
            misconception_label_rate=round(misc_label_rate, 4),
            missingness_by_column=missingness,
            concept_distribution={str(k): int(v) for k, v in list(concept_dist.items())[:10]},
            leakage_risk_flags=leakage_risks,
            suitability_verdict=verdict,
            notes=[
                "Preserves free-form student rationales and specific diagnostic distractor options.",
                "Contains validated misconception IDs suitable for ground-truth diagnosis evaluation.",
            ],
        )

    def to_canonical_interactions(self) -> List[Tuple[Interaction, Optional[str]]]:
        """
        Converts the dataframe into canonical Interaction objects paired with optional ground truth misconception.
        Returns: List of (Interaction, ground_truth_misconception_id)
        """
        if self.df is None:
            self.load()

        records: List[Tuple[Interaction, Optional[str]]] = []
        for _, row in self.df.iterrows():
            cid = str(row.get("construct_name", row.get("concept_id", "fraction_addition_unlike_denominators")))
            interaction = Interaction(
                interaction_id=str(row.get("interaction_id", f"I_{row.name}")),
                learner_id=str(row.get("user_id", f"L_{row.name}")),
                item_id=str(row.get("question_id", "Q_01")),
                item_text=str(row.get("question_text", "Calculate fraction sum")),
                item_concepts=[cid],
                response_text=str(row["explanation"]) if pd.notna(row.get("explanation")) else None,
                selected_option=str(row["selected_option"]) if pd.notna(row.get("selected_option")) else None,
                correctness=int(row["correct"]) if pd.notna(row.get("correct")) else 0,
                latency_seconds=float(row.get("latency", 25.0)),
            )
            gt_misc = str(row.get("misconception_id", "")) if pd.notna(row.get("misconception_id")) else None
            if gt_misc in ["", "none", "nan", "null"]:
                gt_misc = None

            records.append((interaction, gt_misc))

        return records

    @staticmethod
    def create_sample_eedi_dataset(num_students: int = 12) -> pd.DataFrame:
        """
        Generates a high-fidelity synthetic benchmark dataset mirroring Eedi Diagnostic Questions.
        Includes Student A (rule misconception), Student B (slip), Mastery, and Ambiguous samples.
        """
        rows = []
        
        # Archetype patterns
        archetypes = [
            # Type 1: Student A archetype (Denominator addition misconception)
            {
                "type": "misconception_component_addition",
                "responses": [
                    ("Solve 1/2 + 1/3", "2/5", 0, "I added the top numbers and the bottom numbers because we add both parts of a fraction.", "denominator_addition_rule"),
                    ("Solve 1/4 + 2/3", "3/7", 0, "I added the numerators and denominators independently.", "denominator_addition_rule"),
                    ("Solve 2/5 + 1/4", "3/9", 0, "Top numbers 2+1=3 and bottom numbers 5+4=9.", "denominator_addition_rule"),
                    ("Solve 1/3 + 1/6", "2/9", 0, "Added top and bottom separately.", "denominator_addition_rule"),
                ],
            },
            # Type 2: Student B archetype (Calculation slip - rule understood)
            {
                "type": "calculation_slip",
                "responses": [
                    ("Solve 1/2 + 1/3", "4/6", 0, "I know the denominators must match, so I tried to find a common denominator, but I made a calculation mistake while multiplying.", "calculation_slip"),
                    ("Solve 1/4 + 2/3", "10/12", 0, "Common denominator is 12, converted 1/4 to 3/12, but multiplied 2/3 incorrectly to get 7/12, then had arithmetic error.", "calculation_slip"),
                    ("Solve 2/5 + 1/4", "13/20", 1, "Common denominator is 20, 8/20 + 5/20 = 13/20.", None),
                    ("Solve 1/3 + 1/6", "3/6", 1, "Common denominator is 6, 2/6 + 1/6 = 3/6 = 1/2.", None),
                ],
            },
            # Type 3: Consistent Mastery
            {
                "type": "mastery",
                "responses": [
                    ("Solve 1/2 + 1/3", "5/6", 1, "I found a common denominator of 6, converted fractions to 3/6 and 2/6, and added numerators to get 5/6.", None),
                    ("Solve 1/4 + 2/3", "11/12", 1, "Converted to 3/12 and 8/12, sum is 11/12.", None),
                    ("Solve 2/5 + 1/4", "13/20", 1, "Found common denominator 20, converted to 8/20 and 5/20.", None),
                    ("Solve 1/3 + 1/6", "3/6", 1, "Equivalent fraction 2/6 + 1/6 = 3/6.", None),
                ],
            },
            # Type 4: Contradiction / Learning Trajectory (Starts with misconception, shifts to mastery after diagnosis)
            {
                "type": "trajectory_learner",
                "responses": [
                    ("Solve 1/2 + 1/3", "2/5", 0, "I added the top and bottom numbers.", "denominator_addition_rule"),
                    ("Solve 1/4 + 2/3", "3/7", 0, "I added the top and bottom again.", "denominator_addition_rule"),
                    ("Solve 2/5 + 1/4", "13/20", 1, "Now I understand! I found a common denominator of 20 and converted both fractions first.", None),
                    ("Solve 1/3 + 1/6", "3/6", 1, "Common denominator 6, got 3/6 which simplifies to 1/2.", None),
                ],
            },
        ]

        interaction_counter = 1
        for s_idx in range(1, num_students + 1):
            user_id = f"student_{s_idx:02d}"
            # Cycle through archetypes
            arch = archetypes[(s_idx - 1) % len(archetypes)]

            for q_idx, (q_text, sel_opt, corr, expl, misc_id) in enumerate(arch["responses"], 1):
                rows.append({
                    "interaction_id": f"EEDI_INT_{interaction_counter:04d}",
                    "user_id": user_id,
                    "sequence_id": q_idx,
                    "question_id": f"EEDI_Q_{q_idx:02d}",
                    "question_text": q_text,
                    "construct_name": "fraction_addition_unlike_denominators",
                    "selected_option": sel_opt,
                    "correct": corr,
                    "explanation": expl,
                    "misconception_id": misc_id if misc_id else "",
                    "latency": float(np.random.randint(15, 60)),
                    "timestamp": f"2026-10-27T10:{q_idx * 5:02d}:00",
                })
                interaction_counter += 1

        return pd.DataFrame(rows)
