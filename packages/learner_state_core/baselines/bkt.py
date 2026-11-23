from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import roc_auc_score, accuracy_score, brier_score_loss


# Define the offline correctness-only Bayesian Knowledge Tracing baseline.
class ClassicalBKTBaseline:
    """
    Wrapper around pyBKT (Bayesian Knowledge Tracing).
    Represents Baseline 1 (Correctness-Only Knowledge Tracing).
    """

    # Keep a reproducible random seed for offline fitting.
    def __init__(self, seed: int = 42):
        # Store the seed passed to pyBKT.
        self.seed = seed
        # Leave the baseline untrained until fit is called.
        self.model = None

    # Fit a classical knowledge tracing model from ordered learner correctness rows.
    def fit(self, train_df: pd.DataFrame, skills: str = ".*", skill_name: Optional[str] = None) -> Any:
        """
        Fits a standard BKT model using pyBKT.
        train_df must contain columns: ['user_id', 'skill_name', 'correct']
        skills / skill_name: Regex pattern for skill names to fit (defaults to '.*' for all skills).
        """
        # Determine which skills the fitting routine should include.
        target_skills = skill_name if skill_name is not None else skills
        # Give fitting failures a useful baseline-specific error.
        try:
            # Load pyBKT only when the offline fitting method runs.
            from pyBKT.models import Model
            # Initialize a seeded model with three fitting starts and sequential execution.
            self.model = Model(seed=self.seed, num_fits=3, parallel=False)
            # Fit the configured skills to the supplied training rows.
            self.model.fit(data=train_df, skills=target_skills)
            # Expose the fitted model to the experiment.
            return self.model
        # Catch a model setup or fitting failure.
        except Exception as e:
            # Raise a fitting error for the experiment runner.
            raise RuntimeError(f"Failed to fit pyBKT model: {e}")

    # Score a fitted correctness model on supplied evaluation rows.
    def evaluate(self, test_df: pd.DataFrame, skills: str = ".*", skill_name: Optional[str] = None) -> Dict[str, float]:
        """
        Evaluates the fitted BKT model on test data, computing AUC, ACC, Brier score, and ECE.
        """
        # Require a trained model before predictions.
        if self.model is None:
            # Return a clear error when the fit step was skipped.
            raise ValueError("Model must be fitted before evaluation.")

        # Ask pyBKT for per-response correctness predictions.
        preds_df = self.model.predict(data=test_df)
        
        # In pyBKT output, correct_predictions contains the predicted probability of correct answer
        # Read the observed correctness labels.
        y_true = preds_df["correct"].values
        # Column name in pyBKT is typically 'correct_predictions'
        # Verify that prediction probabilities were produced.
        if "correct_predictions" not in preds_df.columns:
            # Reject an incompatible pyBKT output shape.
            raise ValueError("pyBKT output does not contain correct_predictions")
        # Identify the column containing correctness probabilities.
        pred_col = "correct_predictions"
        # Read the predicted probabilities for the matched outcomes.
        y_prob = preds_df[pred_col].values

        # Remove NaNs if any
        # Find rows with both the label and prediction present.
        valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
        # Keep only valid observed labels.
        y_true = y_true[valid_mask]
        # Apply the same row mask to predictions.
        y_prob = y_prob[valid_mask]

        # Check that there are evaluated rows and all probabilities are in range.
        if len(y_true) == 0 or ((y_prob < 0) | (y_prob > 1)).any():
            # Stop rather than compute scores for invalid data.
            raise ValueError("Evaluation requires nonempty valid probability predictions")
        # Check whether both correct and incorrect outcomes occur.
        if len(np.unique(y_true)) < 2:
            # Leave AUC unavailable when ranks cannot distinguish two classes.
            auc = None  # ROC AUC is undefined when the target has one class.
        # With two classes, compute the ordinary ROC AUC.
        else:
            # Compare probability ranking against the observed outcome labels.
            auc = float(roc_auc_score(y_true, y_prob))

        # Calculate accuracy from the classification at a probability threshold of 0.5.
        acc = float(accuracy_score(y_true, (y_prob >= 0.5).astype(int)))
        # Calculate the mean squared probability error with the Brier score.
        brier = float(brier_score_loss(y_true, y_prob))

        # Expected Calibration Error (ECE) with 10 bins
        # Measure calibration error across ten bins.
        ece = self.compute_ece(y_true, y_prob, n_bins=10)

        # Return the offline comparison metrics.
        return {
            # Round AUC when it could be calculated.
            "auc": round(auc, 4) if auc is not None else None,
            # Return the rounded accuracy.
            "accuracy": round(acc, 4),
            # Return the rounded Brier score.
            "brier_score": round(brier, 4),
            # Return the rounded calibration error.
            "ece": round(ece, 4),
        }

    # Calibration can be calculated without an existing fitted model instance.
    @staticmethod
    # Measure the weighted gap between observed outcome rates and average predictions.
    def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Computes Expected Calibration Error."""
        # Check that labels align with predictions and the bin count is valid.
        if n_bins < 1 or len(y_true) != len(y_prob):
            # Reject inconsistent evaluation input.
            raise ValueError("ECE needs matching arrays and a positive bin count")
        # Validate the predicted probabilities before binning.
        if not np.isfinite(y_prob).all() or ((y_prob < 0) | (y_prob > 1)).any():
            # Reject non-finite values and probabilities outside the unit interval.
            raise ValueError("ECE requires finite probability scores in [0,1]")
        # Make evenly spaced bins covering the probability range.
        bin_limits = np.linspace(0, 1, n_bins + 1)
        # Start the weighted calibration discrepancy at zero.
        ece = 0.0
        # Count the observations used to weight bins.
        n_total = len(y_true)
        # Handle an empty input when this helper is called directly.
        if n_total == 0:
            # Return zero total contribution when there are no observations.
            return 0.0

        # Walk each bin in order.
        for i in range(n_bins):
            # Read its lower and upper probability bounds.
            bin_min, bin_max = bin_limits[i], bin_limits[i + 1]
            # Put the upper endpoint in the final bin so a prediction of one is counted.
            in_bin = (y_prob >= bin_min) & (y_prob < bin_max if i < n_bins - 1 else y_prob <= bin_max)
            # Count the observations belonging to this bin.
            n_in_bin = np.sum(in_bin)
            # Skip empty bins.
            if n_in_bin > 0:
                # Measure the actual fraction of correct outcomes in the current bin.
                acc_bin = np.mean(y_true[in_bin])
                # Calculate the mean forecast for that bin.
                conf_bin = np.mean(y_prob[in_bin])
                # Add the absolute difference weighted by the fraction of observations in the bin.
                ece += (n_in_bin / n_total) * abs(acc_bin - conf_bin)

        # Return the resulting Expected Calibration Error as a plain float.
        return float(ece)
