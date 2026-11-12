"""Fast content feature extraction (Phase 5 / 5b)."""

from streamenc.features.analyze import (
    AnalysisResult,
    analyze_video,
    feature_agreement,
)
from streamenc.features.difficulty import DifficultyEstimate, difficulty_score
from streamenc.features.extract import FEATURE_NAMES, FeatureVector

PHASE = 5

__all__ = [
    "PHASE",
    "FEATURE_NAMES",
    "AnalysisResult",
    "DifficultyEstimate",
    "FeatureVector",
    "analyze_video",
    "difficulty_score",
    "feature_agreement",
]
