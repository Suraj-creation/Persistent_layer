from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss


class ClassicalBKTBaseline:
    """
    Wrapper around pyBKT (Bayesian Knowledge Tracing).
    Represents Baseline 1 (Correctness-Only Knowledge Tracing).
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.model = None

    def fit(self, train_df: pd.DataFrame, skills: str = ".*", skill_name: Optional[str] = None) -> Any:
        """
        Fits a standard BKT model using pyBKT.
        train_df must contain columns: ['user_id', 'skill_name', 'correct']
        skills / skill_name: Regex pattern for skill names to fit (defaults to '.*' for all skills).
        """
        target_skills = skill_name if skill_name is not None else skills
        try:
            from pyBKT.models import Model
            self.model = Model(seed=self.seed, num_fits=3)
            self.model.fit(data=train_df, skills=target_skills)
            return self.model
        except Exception as e:
            raise RuntimeError(f"Failed to fit pyBKT model: {e}")

    def evaluate(self, test_df: pd.DataFrame, skills: str = ".*", skill_name: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluates the fitted BKT model on test data, computing AUC, ACC, Brier score, and ECE.
        """
        if self.model is None:
            raise ValueError("Model must be fitted before evaluation.")

        preds_df = self.model.predict(data=test_df)
        
        # In pyBKT output, correct_predictions contains the predicted probability of correct answer
        y_true = preds_df["correct"].values
        # Column name in pyBKT is typically 'correct_predictions'
        pred_col = "correct_predictions" if "correct_predictions" in preds_df.columns else preds_df.columns[-1]
        y_prob = preds_df[pred_col].values

        # Remove NaNs if any
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        y_true = y_true[valid_mask]
        y_prob = y_prob[valid_mask]

        if len(np.unique(y_true)) < 2:
            auc = 0.50
        else:
            auc = float(roc_auc_score(y_true, y_prob))

        acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
        brier = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error (ECE) with 10 bins
        ece = self.compute_ece(y_true, y_prob, n_bins=10)

        return {
            "auc": round(auc, 4),
            "accuracy": round(acc, 4),
            "brier_score": round(brier, 4),
            "ece": round(ece, 4),
        }

    @staticmethod
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error."""
        bin_limits = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        n_total = len(y_true)
        if n_total == 0:
            return 0.0

        for i in range(n_bins):
            bin_min, bin_max = bin_limits[i], bin_limits[i + 1]
            in_bin = (y_prob >= bin_min) & (y_prob < bin_max if i < n_bins - 1 else y_prob <= bin_max)
            n_in_bin = np.sum(in_bin)
            if n_in_bin > 0:
                acc_bin = np.mean(y_true[in_bin])
                conf_bin = np.mean(y_prob[in_bin])
                ece += (n_in_bin / n_total) * abs(acc_bin - conf_bin)

        return float(ece)
