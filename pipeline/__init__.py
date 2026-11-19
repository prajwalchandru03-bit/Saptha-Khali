"""
TruthLens Multimodal Forensic Pipeline
"""

from pipeline.ingest import IngestPipeline, IngestResult, compute_sha256
from pipeline.visual import (
    VisualAnalyzer,
    VisualForensicsResult,
    ExifAnalysisResult,
    ElaResult,
    FramePrediction
)
from pipeline.audio import (
    AudioAnalyzer,
    AudioForensicsResult,
    AudioChunkResult,
    TranscriptSegment
)
from pipeline.cross_modal import (
    CrossModalAnalyzer,
    CrossModalResult,
    LipSyncResult,
    ClipCaptionResult,
    SceneMismatchResult
)
from pipeline.correlator import (
    Correlator,
    ForensicReport,
    FlaggedEvidenceItem,
    CrossModalFindings,
    ConfidenceLevel
)

__all__ = [
    "IngestPipeline",
    "IngestResult",
    "compute_sha256",
    "VisualAnalyzer",
    "VisualForensicsResult",
    "ExifAnalysisResult",
    "ElaResult",
    "FramePrediction",
    "AudioAnalyzer",
    "AudioForensicsResult",
    "AudioChunkResult",
    "TranscriptSegment",
    "CrossModalAnalyzer",
    "CrossModalResult",
    "LipSyncResult",
    "ClipCaptionResult",
    "SceneMismatchResult",
    "Correlator",
    "ForensicReport",
    "FlaggedEvidenceItem",
    "CrossModalFindings",
    "ConfidenceLevel",
]
