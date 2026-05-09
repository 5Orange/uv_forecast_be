from __future__ import annotations

import numpy as np
import pandas as pd


SKIN_TYPE_MULTIPLIER = {
    1: 2.5,
    2: 3.0,
    3: 4.0,
    4: 5.0,
    5: 8.0,
    6: 15.0,
}

UV_CATEGORY_REPRESENTATIVE = {
    # Mirrors WHO-ish bins used in `FeatureEngineer.add_targets()` / `config.UV_CATEGORIES`
    # uv_category bins in dataset: [-0.1..2, 3..5, 6..7, 8..10, 11+]
    0: 1.0,   # low
    1: 4.0,   # moderate
    2: 6.5,   # high
    3: 9.0,   # very_high
    4: 11.5,  # extreme
}


def uv_index_from_category(uv_category: float | int | np.ndarray) -> float | np.ndarray:
    """
    Convert categorical WHO UV warnings (0..4) into a representative UV index.
    Used only when `uv_index` is not available.
    """
    if isinstance(uv_category, np.ndarray):
        out = np.empty_like(uv_category, dtype=float)
        for k, v in UV_CATEGORY_REPRESENTATIVE.items():
            out[uv_category == k] = float(v)
        # If any unexpected values appear, fall back to 1.0
        out[np.isnan(out)] = 1.0
        return out
    return float(UV_CATEGORY_REPRESENTATIVE.get(int(uv_category), 1.0))


def estimate_safe_minutes(uv_index: float | np.ndarray, skin_type: int) -> float | np.ndarray:
    """
    Estimate safe exposure time in minutes.

    Formula: safe_minutes = (200 * multiplier) / (3 * max(1, uv_index))
    Source: ScanSkinAI (http://www.scanskinai.com/safe-sun-exposure-time)
    """
    if skin_type not in SKIN_TYPE_MULTIPLIER:
        raise ValueError(f"skin_type must be in {sorted(SKIN_TYPE_MULTIPLIER.keys())}")

    multiplier = SKIN_TYPE_MULTIPLIER[skin_type]
    uv_factor = np.maximum(1.0, np.asarray(uv_index, dtype=float))
    return (200.0 * multiplier) / (3.0 * uv_factor)


def safe_minutes_from_category(uv_category: float | int | np.ndarray, skin_type: int) -> float | np.ndarray:
    """Convenience wrapper: uv_category -> representative uv_index -> safe minutes."""
    uv_index = uv_index_from_category(uv_category)
    return estimate_safe_minutes(uv_index=uv_index, skin_type=skin_type)


def add_safe_flags(
    df: pd.DataFrame,
    *,
    skin_type: int = 3,
    activity_duration_minutes: float = 60.0,
    uv_category_col: str = "uv_category",
    uv_index_col: str | None = None,
    out_prefix: str = "",
) -> pd.DataFrame:
    """
    Add:
      - {out_prefix}safe_minutes
      - {out_prefix}is_safe (True if safe_minutes >= activity_duration_minutes)
    """
    if uv_index_col is None:
        if uv_category_col not in df.columns:
            raise KeyError(f"Missing `{uv_category_col}` column in df")
        safe_minutes = safe_minutes_from_category(df[uv_category_col].values, skin_type=skin_type)
    else:
        if uv_index_col not in df.columns:
            raise KeyError(f"Missing `{uv_index_col}` column in df")
        safe_minutes = estimate_safe_minutes(df[uv_index_col].values, skin_type=skin_type)

    safe_minutes_col = f"{out_prefix}safe_minutes"
    is_safe_col = f"{out_prefix}is_safe"
    df = df.copy()
    df[safe_minutes_col] = safe_minutes
    df[is_safe_col] = df[safe_minutes_col] >= activity_duration_minutes
    return df


def safe_time_ranges(
    df: pd.DataFrame,
    *,
    time_col: str = "timestamp",
    is_safe_col: str = "is_safe",
    max_gap_hours: float = 1.0,
) -> list[dict]:
    """
    Convert boolean safe flags into contiguous time ranges.
    Assumes `timestamp` is hourly (or close enough).
    """
    if df.empty:
        return []
    if time_col not in df.columns or is_safe_col not in df.columns:
        raise KeyError("Missing required columns for safe_time_ranges()")

    tmp = df[[time_col, is_safe_col]].dropna().sort_values(time_col).copy()
    tmp[time_col] = pd.to_datetime(tmp[time_col])

    # Keep only safe rows for range building.
    safe = tmp[tmp[is_safe_col] == True]
    if safe.empty:
        return []

    ranges: list[dict] = []
    start = safe.iloc[0][time_col]
    prev = start

    for _, row in safe.iloc[1:].iterrows():
        t = row[time_col]
        gap = (t - prev).total_seconds() / 3600.0
        if gap <= max_gap_hours + 1e-9:
            prev = t
            continue
        # close previous range
        ranges.append({"start": start.isoformat(), "end": prev.isoformat()})
        start = t
        prev = t

    ranges.append({"start": start.isoformat(), "end": prev.isoformat()})
    return ranges

