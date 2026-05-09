from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from src.config import LOCATIONS
from src.recommendation.safe_time_policy import add_safe_flags


Activity = Literal["beach", "sightseeing", "indoor"]


def _contiguous_time_ranges(
    df_sorted: pd.DataFrame,
    *,
    time_col: str,
    max_gap_hours: float,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if df_sorted.empty:
        return []

    start = pd.to_datetime(df_sorted.iloc[0][time_col])
    prev = start
    ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    for _, row in df_sorted.iloc[1:].iterrows():
        t = pd.to_datetime(row[time_col])
        gap = (t - prev).total_seconds() / 3600.0
        if gap <= max_gap_hours + 1e-9:
            prev = t
            continue
        ranges.append((start, prev))
        start = t
        prev = t

    ranges.append((start, prev))
    return ranges


def recommend_for_location(
    df: pd.DataFrame,
    *,
    location_id: str,
    skin_type: int = 3,
    activity_duration_minutes: float = 60.0,
    uv_index_col: str = "uv_predicted",
    time_col: str = "timestamp",
    max_gap_hours: float = 1.0,
    top_k: int = 5,
) -> dict:
    """
    Rule-based recommendations for one location.

    Expected columns in `df`:
      - {time_col}, location_id
      - continuous UV index column: `uv_index_col` (from regression model)
      - outdoor suitability: beach_suitability, sightseeing_suitability, outdoor_suitability_enhanced
      - indoor preference: indoor_preference
      - weather/heat: is_raining, cloud_opacity, heat_warning, extreme_heat_warning
    """
    if location_id not in df["location_id"].unique():
        return {"location_id": location_id, "recommendations": []}

    loc_type = LOCATIONS[location_id]["type"]
    dfl = df[df["location_id"] == location_id].sort_values(time_col).copy()

    # Compute safe minutes directly from continuous UV index (more accurate than category mapping)
    dfl = add_safe_flags(
        dfl,
        skin_type=skin_type,
        activity_duration_minutes=activity_duration_minutes,
        uv_index_col=uv_index_col,
        out_prefix="",
    )
    # safe_minutes, is_safe

    safe_ratio = (dfl["safe_minutes"] / float(activity_duration_minutes)).clip(lower=0.0, upper=3.0)

    # Weather penalty (outdoor only)
    is_raining = dfl.get("is_raining", 0).astype(float).fillna(0.0)
    cloud_opacity = dfl.get("cloud_opacity", 0.0).astype(float).fillna(0.0)
    cloud_frac = (cloud_opacity / 100.0).clip(0.0, 1.0)

    rain_ok_factor = np.where(is_raining > 0.0, 0.6, 1.0)
    cloud_ok_factor = 1.0 - 0.4 * cloud_frac
    outdoor_weather_penalty = rain_ok_factor * cloud_ok_factor

    heat_outdoor_factor = np.where(dfl.get("extreme_heat_warning", 0).fillna(0) > 0, 0.6, 1.0)
    heat_indoor_factor = np.where(dfl.get("extreme_heat_warning", 0).fillna(0) > 0, 1.15, 1.0)

    beach_suit = dfl.get("beach_suitability", 0).astype(float).fillna(0.0)
    sight_suit = dfl.get("sightseeing_suitability", 0).astype(float).fillna(0.0)
    outdoor_enh = dfl.get("outdoor_suitability_enhanced", 1).astype(float).fillna(1.0)
    indoor_pref = dfl.get("indoor_preference", 0).astype(float).fillna(0.0)

    # Score definitions: scaled by (safe_ratio) but outdoor only when safe.
    # If it's not a coastal location, beach is suppressed.
    not_coastal = loc_type != "coastal"

    beach_score = beach_suit * outdoor_weather_penalty * heat_outdoor_factor * outdoor_enh * safe_ratio
    sightseeing_score = sight_suit * outdoor_weather_penalty * heat_outdoor_factor * outdoor_enh * safe_ratio

    # Enforce safety policy for outdoor activities:
    beach_score = np.where(dfl["is_safe"].values, beach_score, 0.0)
    sightseeing_score = np.where(dfl["is_safe"].values, sightseeing_score, 0.0)
    if not_coastal:
        beach_score[:] = 0.0

    indoor_score = indoor_pref * heat_indoor_factor

    dfl["beach_score"] = beach_score
    dfl["sightseeing_score"] = sightseeing_score
    dfl["indoor_score"] = indoor_score

    best_idx = np.argmax(
        np.vstack([dfl["beach_score"].values, dfl["sightseeing_score"].values, dfl["indoor_score"].values]),
        axis=0,
    )
    best_activity: list[Activity] = [  # order must match stacking above
        "beach" if i == 0 else "sightseeing" if i == 1 else "indoor" for i in best_idx.tolist()
    ]
    dfl["selected_activity"] = best_activity

    # Build contiguous ranges per selected activity.
    activity_score_col = {
        "beach": "beach_score",
        "sightseeing": "sightseeing_score",
        "indoor": "indoor_score",
    }

    recommendations: list[dict] = []
    for activity in ["beach", "sightseeing", "indoor"]:
        dfa = dfl[dfl["selected_activity"] == activity]
        if dfa.empty:
            continue
        ranges = _contiguous_time_ranges(dfa, time_col=time_col, max_gap_hours=max_gap_hours)
        for start, end in ranges:
            mask = (pd.to_datetime(dfa[time_col]) >= start) & (pd.to_datetime(dfa[time_col]) <= end)
            mean_score = float(dfa.loc[mask, activity_score_col[activity]].mean())
            recommendations.append(
                {
                    "activity": activity,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "mean_score": mean_score,
                }
            )

    # Return top_k intervals by mean_score.
    recommendations = sorted(recommendations, key=lambda x: x["mean_score"], reverse=True)[:top_k]
    return {
        "location_id": location_id,
        "location_type": loc_type,
        "skin_type": skin_type,
        "activity_duration_minutes": activity_duration_minutes,
        "recommendations": recommendations,
    }


def recommend_all_locations(
    df: pd.DataFrame,
    *,
    skin_type: int = 3,
    activity_duration_minutes: float = 60.0,
    uv_index_col: str = "uv_predicted",
    top_k: int = 5,
) -> list[dict]:
    outputs: list[dict] = []
    for location_id in sorted(df["location_id"].unique().tolist()):
        outputs.append(
            recommend_for_location(
                df,
                location_id=location_id,
                skin_type=skin_type,
                activity_duration_minutes=activity_duration_minutes,
                uv_index_col=uv_index_col,
                top_k=top_k,
            )
        )
    return outputs

