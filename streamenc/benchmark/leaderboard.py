"""Aggregate experiment rows into competition-oriented leaderboards."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from streamenc.score import speed_multiplier, speed_score


def experiments_frame(rows: Iterable[dict]) -> pd.DataFrame:
    df = pd.DataFrame(list(rows))
    if df.empty:
        return df
    return df


def config_leaderboard(df: pd.DataFrame, *, split: str | None = None) -> pd.DataFrame:
    """Rank configs by estimated competition score on a split.

    Uses mean clip_score among validated rows, zeros for gate failures,
    and speed from mean encode_seconds scaled to 3428 frames via mean_spf.
    """
    if df.empty:
        return df
    work = df.copy()
    if split:
        work = work[work["split"] == split]
    # Only rows that attempted full validation
    work = work[work["validation_passed"].notna()]
    if work.empty:
        return work

    rows = []
    for config_id, g in work.groupby("config_id"):
        n = len(g)
        passed = int((g["validation_passed"] == True).sum())  # noqa: E712
        # Zero-score failed clips
        scores = []
        for _, r in g.iterrows():
            if r["validation_passed"] and pd.notna(r.get("clip_score")):
                scores.append(float(r["clip_score"]))
            else:
                scores.append(0.0)
        quality = sum(scores) / n if n else 0.0
        mean_encode_s = float(g["encode_seconds"].mean())
        mean_frames = float(g["frames"].mean())
        # Extrapolate to competition inserted frame count using observed spf
        spf = mean_encode_s / max(mean_frames, 1.0)
        total_runtime = spf * 3428.0
        speed = speed_score(total_runtime, 3428.0)
        est = quality * speed_multiplier(speed)
        rows.append(
            {
                "config_id": config_id,
                "codec": g["codec"].iloc[0],
                "n_clips": n,
                "n_passed": passed,
                "pass_rate": passed / n,
                "mean_vmaf": float(g["vmaf"].dropna().mean()) if g["vmaf"].notna().any() else None,
                "mean_clip_score": quality,
                "mean_encode_seconds": mean_encode_s,
                "spf": spf,
                "extrapolated_runtime_s": total_runtime,
                "speed_score": speed,
                "speed_multiplier": speed_multiplier(speed),
                "estimated_final_score": est,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["estimated_final_score", "mean_clip_score", "pass_rate"],
        ascending=False,
    ).reset_index(drop=True)


def stage1_rank(df: pd.DataFrame) -> pd.DataFrame:
    """Rank stage-1 encode screens by compliance and speed (no VMAF)."""
    if df.empty:
        return df
    rows = []
    for config_id, g in df.groupby("config_id"):
        n = len(g)
        ok = int(g["encode_ok"].sum())
        rows.append(
            {
                "config_id": config_id,
                "codec": g["codec"].iloc[0],
                "n_clips": n,
                "encode_pass_rate": ok / n if n else 0.0,
                "mean_encode_seconds": float(g["encode_seconds"].mean()),
                "mean_avg_kbps_util": float(
                    (g["avg_kbps"] / g["target_kbps"]).dropna().mean()
                )
                if g["avg_kbps"].notna().any()
                else None,
            }
        )
    out = pd.DataFrame(rows)
    return out.sort_values(
        ["encode_pass_rate", "mean_encode_seconds"],
        ascending=[False, True],
    ).reset_index(drop=True)
