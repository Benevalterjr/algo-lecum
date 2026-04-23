"""Funções de alocação de portfólio com restrição de exposição."""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_allocation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["Conviction_Score"] = out["Confidence_Prob"] * out["Energy_Gap"]

    def assign_weight(row: pd.Series) -> float:
        if row["Confidence_Prob"] > 0.9 and row["Energy_Gap"] > 2.0:
            return 0.40
        if row["Confidence_Prob"] > 0.7 and row["Energy_Gap"] > 0.5:
            return 0.25
        if row["Confidence_Prob"] > 0.6:
            return 0.15
        return 0.00

    out["Suggested_Weight"] = out.apply(assign_weight, axis=1)
    total = float(out["Suggested_Weight"].sum())
    out["Normalized_Weight"] = (
        out["Suggested_Weight"] / total if total > 1.0 else out["Suggested_Weight"]
    )
    return out.sort_values(by="Conviction_Score", ascending=False).reset_index(drop=True)


def professional_allocation(
    df: pd.DataFrame,
    max_exposure: float = 0.85,
    temperature: float = 1.5,
) -> pd.DataFrame:
    out = df.copy()
    scores = out["Conviction_Score"].to_numpy(dtype=float)
    exp_scores = np.exp(scores / temperature)
    softmax_weights = exp_scores / exp_scores.sum()

    out["Final_Proportional_Weight"] = softmax_weights * max_exposure
    out["Strategic_Impact"] = out["Conviction_Score"] * out["Final_Proportional_Weight"]
    return out.sort_values(by="Final_Proportional_Weight", ascending=False).reset_index(drop=True)
