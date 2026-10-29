from __future__ import annotations

from typing import List, Tuple
import numpy as np
import pandas as pd


class LeakageSafeSplitter:
    """
    Generates student-level and chronological splits to prevent data leakage in KT and diagnosis.
    """

    @staticmethod
    def student_split(
        df: pd.DataFrame,
        student_col: str = "user_id",
        test_ratio: float = 0.25,
        seed: int = 42,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits dataset by student ID: all interactions for a student belong strictly to train OR test.
        Prevents student-identity leakage.
        """
        np.random.seed(seed)
        unique_students = np.array(df[student_col].unique())
        np.random.shuffle(unique_students)

        split_idx = int(len(unique_students) * (1.0 - test_ratio))
        train_students = set(unique_students[:split_idx])
        test_students = set(unique_students[split_idx:])

        train_df = df[df[student_col].isin(train_students)].copy().reset_index(drop=True)
        test_df = df[df[student_col].isin(test_students)].copy().reset_index(drop=True)

        return train_df, test_df

    @staticmethod
    def student_k_fold(
        df: pd.DataFrame,
        student_col: str = "user_id",
        k: int = 5,
        seed: int = 42,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generates k leakage-safe folds partitioned at the student level.
        Returns: List of (train_df, test_df) for each fold.
        """
        np.random.seed(seed)
        unique_students = np.array(df[student_col].unique())
        np.random.shuffle(unique_students)

        student_folds = np.array_split(unique_students, k)
        folds = []

        for i in range(k):
            test_students = set(student_folds[i])
            train_students = set(unique_students) - test_students

            train_df = df[df[student_col].isin(train_students)].copy().reset_index(drop=True)
            test_df = df[df[student_col].isin(test_students)].copy().reset_index(drop=True)
            folds.append((train_df, test_df))

        return folds
