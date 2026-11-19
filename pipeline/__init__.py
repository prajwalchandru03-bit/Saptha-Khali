"""
TruthLens Multimodal Forensic Pipeline
"""

from pipeline.ingest import IngestPipeline, IngestResult, IngestionError, compute_sha256
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
    ConfidenceLevel,
    AssessmentItem,
    AssessmentLevel,
    MediaOriginAssessment
)
from pipeline.compare import (
    compare_media,
    ComparisonResult,
    LocalForensicsSummary
)
from pipeline.assistant import (
    investigation_assistant,
    AssistantResponse
)
from pipeline.text_forensics import (
    analyze_text_forensics,
    TextForensicsResult
)

__all__ = [
    "IngestPipeline",
    "IngestResult",
    "IngestionError",
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
    "AssessmentItem",
    "AssessmentLevel",
    "MediaOriginAssessment",
    "compare_media",
    "ComparisonResult",
    "LocalForensicsSummary",
    "investigation_assistant",
    "AssistantResponse",
    "analyze_text_forensics",
    "TextForensicsResult",
]
