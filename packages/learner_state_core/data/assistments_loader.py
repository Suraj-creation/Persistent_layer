from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np

from ..evidence.models import Interaction
from .schemas import DatasetAuditReport


class FoundationalAssistLoader:
    """
    Data connector for FoundationalASSIST dataset (Hugging Face / ASSISTments 2026).
    Audits whether actual responses are free-form explanations or numeric answers.
    """

    def __init__(self, data_path: Optional[Union[str, Path]] = None):
        self.data_path = Path(data_path) if data_path else None
        self.df: Optional[pd.DataFrame] = None

    def load(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Loads data from file or returns synthetic benchmark dataframe."""
        if df is not None:
            self.df = df
        elif self.data_path and self.data_path.exists():
            if self.data_path.suffix == ".csv":
                self.df = pd.read_csv(self.data_path)
            elif self.data_path.suffix == ".parquet":
                self.df = pd.read_parquet(self.data_path)
            else:
                self.df = pd.read_json(self.data_path)
        else:
            self.df = self.create_sample_foundational_assist_dataset()

        return self.df

    def audit(self) -> DatasetAuditReport:
        """Audits FoundationalASSIST data, specifically checking text availability vs numeric answers."""
        if self.df is None:
            self.load()

        df = self.df
        total_records = len(df)
        total_students = int(df["user_id"].nunique()) if "user_id" in df.columns else 0
        total_items = int(df["problem_id"].nunique()) if "problem_id" in df.columns else 0

        skill_col = "skill_name" if "skill_name" in df.columns else "concept_id"
        total_concepts = int(df[skill_col].nunique()) if skill_col in df.columns else 0
        concept_dist = df[skill_col].value_counts().to_dict() if skill_col in df.columns else {}

        # Check if responses are actual explanations or purely short numeric strings
        ans_col = "answer_text" if "answer_text" in df.columns else "student_response"
        is_free_text = []
        if ans_col in df.columns:
            for val in df[ans_col].dropna():
                s = str(val).strip()
                # If more than 3 words, considered explanation
                is_free_text.append(len(s.split()) >= 3)
            free_text_rate = float(np.mean(is_free_text)) if is_free_text else 0.0
        else:
            free_text_rate = 0.0

        missingness = {col: float(df[col].isna().mean()) for col in df.columns}

        leakage_flags = []
        if "user_id" in df.columns and "sequence_id" not in df.columns:
            leakage_flags.append("TIMESTAMP_MISSING: Relying on index order.")

        verdict = (
            "SUITABLE_FOR_RESPONSE_PREDICTION_AND_PARTIAL_EXPLANATION"
            if free_text_rate >= 0.40
            else "SUITABLE_PRIMARILY_FOR_CORRECTNESS_KT_AND_OPTION_ANALYSIS"
        )

        return DatasetAuditReport(
            dataset_name="FoundationalASSIST (ASSISTments 2026)",
            total_records=total_records,
            total_students=total_students,
            total_items=total_items,
            total_concepts=total_concepts,
            text_availability_rate=round(free_text_rate, 4),
            misconception_label_rate=0.0,  # FoundationalASSIST does not natively supply verified misconception IDs
            missingness_by_column=missingness,
            concept_distribution={str(k): int(v) for k, v in list(concept_dist.items())[:10]},
            leakage_risk_flags=leakage_flags,
            suitability_verdict=verdict,
            notes=[
                "Preserves exact problem stems, answer choices, and student-entered text.",
                "Notice: Does not supply explicit misconception tags; annotation or distractor mapping needed.",
            ],
        )

    def to_canonical_interactions(self) -> List[Interaction]:
        """Converts dataframe to list of canonical Interaction objects."""
        if self.df is None:
            self.load()

        interactions = []
        for _, row in self.df.iterrows():
            interactions.append(
                Interaction(
                    interaction_id=str(row.get("interaction_id", f"FA_{row.name}")),
                    learner_id=str(row.get("user_id", f"STU_{row.name}")),
                    item_id=str(row.get("problem_id", f"PROB_{row.name}")),
                    item_text=str(row.get("problem_text", "")),
                    item_concepts=[str(row.get("skill_name", "fraction_addition"))],
                    response_text=str(row.get("answer_text", "")),
                    selected_option=str(row.get("selected_option", "")) if pd.notna(row.get("selected_option")) else None,
                    correctness=int(row.get("correct", 0)),
                    attempt_number=int(row.get("attempt_count", 1)),
                    hint_used=bool(row.get("hint_count", 0) > 0),
                )
            )
        return interactions

    @staticmethod
    def create_sample_foundational_assist_dataset(num_students: int = 10) -> pd.DataFrame:
        """Generates representative synthetic FoundationalASSIST interactions."""
        rows = []
        problems = [
            ("P_101", "Calculate: 1/2 + 1/4", "fraction_addition_unlike_denominators", "3/4", 1, 0, 1),
            ("P_102", "Calculate: 2/3 + 1/6", "fraction_addition_unlike_denominators", "5/6", 1, 0, 1),
            ("P_103", "Calculate: 1/5 + 2/5", "fraction_addition_like_denominators", "3/5", 1, 0, 1),
            ("P_104", "Calculate: 3/4 - 1/2", "fraction_addition_unlike_denominators", "2/6", 0, 1, 2),
        ]

        counter = 1
        for s in range(1, num_students + 1):
            user_id = f"FA_USER_{s:03d}"
            for p_id, p_text, skill, ans, corr, hints, atts in problems:
                rows.append({
                    "interaction_id": f"FA_INT_{counter:04d}",
                    "user_id": user_id,
                    "problem_id": p_id,
                    "problem_text": p_text,
                    "skill_name": skill,
                    "answer_text": ans if s % 2 == 0 else f"I think the answer is {ans}",
                    "correct": corr,
                    "hint_count": hints,
                    "attempt_count": atts,
                    "sequence_id": counter,
                })
                counter += 1

        return pd.DataFrame(rows)
