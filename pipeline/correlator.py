"""
TruthLens Multimodal Forensic Correlator Module
Sends all extracted signals, transcripts, captions, and top flagged frames to
Gemini API (google-genai) and returns a strict Pydantic JSON forensic report.
"""

import io
import json
import logging
import os
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field
from PIL import Image

logger = logging.getLogger("truthlens.correlator")


class ConfidenceLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"


class ModalityType(str, Enum):
    VISUAL = "visual"
    AUDIO = "audio"
    CROSS_MODAL = "cross_modal"
    METADATA = "metadata"
    TEXT = "text"


class FlaggedEvidenceItem(BaseModel):
    """Specific piece of forensic evidence or anomaly discovered."""
    modality: str = Field(
        description="Modality of evidence: 'visual', 'audio', 'cross_modal', 'metadata', or 'text'"
    )
    timestamp_or_region: str = Field(
        description="Timestamp range (e.g. '00:03-00:05s') or spatial region (e.g. 'ELA heatmap center face', 'Frame 12')"
    )
    indicator: str = Field(
        description="Technical forensic indicator name (e.g. 'ViT Synthetic Artifacts', 'Audio Desync', 'ELA Compression Anomaly')"
    )
    reason: str = Field(
        description="Detailed forensic explanation of what makes this anomalous or authentic"
    )


class CrossModalFindings(BaseModel):
    """Multi-angle cross-modal consistency findings."""
    lip_sync: str = Field(
        description="Forensic assessment of lip-sync correlation (MediaPipe FaceMesh mouth opening vs audio RMS energy)"
    )
    caption_vs_content: str = Field(
        description="Evaluation of social media caption vs visual frame semantics (CLIP ViT)"
    )
    audio_vs_visual: str = Field(
        description="Evaluation of acoustic background scene vs visual scene consistency (AST vs CLIP)"
    )
    transcript_vs_caption: str = Field(
        description="Forensic comparison between spoken speech transcript and post caption to detect contradictions or fabricated context"
    )


class ForensicReport(BaseModel):
    """Strict Pydantic JSON forensic report produced by Gemini correlator."""
    sha256_hash: str = Field(
        description="Cryptographic SHA-256 hash of the analyzed media file for evidence provenance"
    )
    verdict: str = Field(
        description="Overall forensic verdict: 'Likely Authentic', 'Manipulated / Deepfake', 'Out-of-Context Misinformation', or 'Suspicious / Inconclusive'"
    )
    deepfake_probability: float = Field(
        ge=0.0, le=1.0,
        description="Aggregated calculated probability of synthetic manipulation (0.0 = completely genuine, 1.0 = confirmed synthetic/deepfake)"
    )
    confidence_level: ConfidenceLevel = Field(
        description="Confidence level of this forensic determination: Low, Moderate, or High"
    )
    summary: str = Field(
        description="Comprehensive executive summary synthesizing all forensic signals, modalities, and cross-checks"
    )
    flagged_evidence: List[FlaggedEvidenceItem] = Field(
        default_factory=list,
        description="Specific forensic anomalies and indicators discovered across modalities"
    )
    cross_modal_findings: CrossModalFindings = Field(
        description="Detailed cross-modal findings (lip_sync, caption_vs_content, audio_vs_visual, transcript_vs_caption)"
    )
    limitations_and_caveats: List[str] = Field(
        default_factory=list,
        description="Known technical limitations, platform recompression effects, or forensic caveats"
    )
    recommended_human_checks: List[str] = Field(
        default_factory=list,
        description="Actionable forensic verification steps recommended for human OSINT investigators or fact-checkers"
    )


class Correlator:
    """
    Multimodal Correlator that communicates with the Gemini API (google-genai)
    to synthesize signals from all modalities into a strict Pydantic ForensicReport.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        self.client = None

        if self.api_key:
            try:
                from google import genai
                self.client = genai.Client(api_key=self.api_key)
                logger.info(f"Initialized Google GenAI client with model: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize Google GenAI Client: {e}")

    def generate_report(
        self,
        sha256_hash: str,
        ingest_info: Dict[str, Any],
        visual_info: Dict[str, Any],
        audio_info: Dict[str, Any],
        cross_modal_info: Dict[str, Any],
        caption: Optional[str] = None,
        flagged_frame_paths: Optional[List[str]] = None
    ) -> ForensicReport:
        """
        Sends all signals, transcripts, captions, and top flagged frames to Gemini API
        and parses into a strict Pydantic ForensicReport.
        Falls back to local heuristic synthesis if Gemini API key is missing or unavailable.
        """
        # Attempt Gemini API synthesis if client is available
        if self.client:
            try:
                return self._call_gemini(
                    sha256_hash=sha256_hash,
                    ingest_info=ingest_info,
                    visual_info=visual_info,
                    audio_info=audio_info,
                    cross_modal_info=cross_modal_info,
                    caption=caption,
                    flagged_frame_paths=flagged_frame_paths or []
                )
            except Exception as e:
                logger.error(f"Gemini API generation failed: {e}. Falling back to rule-based synthesis.", exc_info=True)

        # Fallback local deterministic synthesis
        return self._local_heuristic_synthesis(
            sha256_hash=sha256_hash,
            ingest_info=ingest_info,
            visual_info=visual_info,
            audio_info=audio_info,
            cross_modal_info=cross_modal_info,
            caption=caption,
            flagged_frame_paths=flagged_frame_paths or []
        )

    def _call_gemini(
        self,
        sha256_hash: str,
        ingest_info: Dict[str, Any],
        visual_info: Dict[str, Any],
        audio_info: Dict[str, Any],
        cross_modal_info: Dict[str, Any],
        caption: Optional[str],
        flagged_frame_paths: List[str]
    ) -> ForensicReport:
        """Invokes google-genai SDK with structured Pydantic schema."""
        from google.genai import types

        # Prepare multimodal contents
        contents: List[Any] = []

        system_instruction = (
            "You are TruthLens Senior Forensic Analyst, an expert in multimodal digital forensics, "
            "deepfake detection, reverse image analysis, speech synthesis detection, and social media disinformation verification. "
            "Analyze the provided forensic telemetry, metadata, model outputs, acoustic-visual sync indicators, and attached frame images. "
            "Correlate all modalities rigorously to generate a strict JSON forensic report adhering strictly to the schema provided. "
            "Provide objective, evidence-backed reasoning with specific timestamps, indicators, and actionable verification steps."
        )

        prompt_payload = {
            "task": "Perform Multimodal Forensic Correlation & Generate Forensic Report",
            "sha256_hash": sha256_hash,
            "social_media_caption": caption or "None provided",
            "ingest_metadata": ingest_info,
            "visual_forensics": visual_info,
            "audio_forensics": audio_info,
            "cross_modal_forensics": cross_modal_info,
        }

        prompt_text = (
            f"{system_instruction}\n\n"
            f"FORENSIC SIGNALS AND TELEMETRY:\n{json.dumps(prompt_payload, indent=2, default=str)}\n\n"
            "Examine any attached flagged frames directly for visual artifacts (warping, blending boundaries, unnatural gaze, lighting mismatch). "
            "Return a complete and strictly valid ForensicReport JSON conforming to the schema."
        )
        contents.append(prompt_text)

        # Attach up to 3 top flagged frame images for visual inspection by Gemini
        for f_path in flagged_frame_paths[:3]:
            try:
                p = Path(f_path)
                if p.exists():
                    with open(p, "rb") as f:
                        img_bytes = f.read()
                    contents.append(
                        types.Part.from_bytes(
                            data=img_bytes,
                            mime_type="image/jpeg"
                        )
                    )
            except Exception as img_err:
                logger.warning(f"Could not load flagged frame for Gemini prompt: {img_err}")

        # Configure structured output using ForensicReport Pydantic model
        config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=ForensicReport,
            temperature=0.2,
        )

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=config,
        )

        if not response.text:
            raise ValueError("Empty response received from Gemini API.")

        report = ForensicReport.model_validate_json(response.text)
        return report

    def _local_heuristic_synthesis(
        self,
        sha256_hash: str,
        ingest_info: Dict[str, Any],
        visual_info: Dict[str, Any],
        audio_info: Dict[str, Any],
        cross_modal_info: Dict[str, Any],
        caption: Optional[str],
        flagged_frame_paths: List[str]
    ) -> ForensicReport:
        """
        Deterministic rule-based forensic aggregator fallback ensuring valid
        ForensicReport output even when offline or without Gemini API key.
        """
        flagged_evidence: List[FlaggedEvidenceItem] = []
        deepfake_scores: List[float] = []

        # 1. Visual Signals
        avg_vis_fake = visual_info.get("average_fake_score", 0.0)
        peak_vis_fake = visual_info.get("peak_fake_score", 0.0)
        deepfake_scores.append(avg_vis_fake)

        if peak_vis_fake >= 0.6:
            flagged_evidence.append(FlaggedEvidenceItem(
                modality=ModalityType.VISUAL.value,
                timestamp_or_region=f"Peak frame ({visual_info.get('peak_frame_timestamp', '00:00s')})",
                indicator="ViT Deepfake Artifacts",
                reason=f"HuggingFace ViT detector flagged facial synthetic characteristics with confidence {peak_vis_fake:.2f}."
            ))

        # ELA findings
        ela = visual_info.get("ela_result") or {}
        ela_score = ela.get("tampering_suspicion_score", 0.0)
        if ela_score > 0.65:
            deepfake_scores.append(ela_score)
            flagged_evidence.append(FlaggedEvidenceItem(
                modality=ModalityType.VISUAL.value,
                timestamp_or_region="Spatial Heatmap Anomaly",
                indicator="Error Level Analysis (ELA) Compression Inconsistency",
                reason=f"High localized variance ({ela.get('error_variance', 0)}) detected, indicating regional splicing or re-compression."
            ))

        # EXIF findings
        exif = visual_info.get("exif_analysis") or {}
        for indicator in exif.get("suspicious_indicators", []):
            if "Software signature" in indicator:
                deepfake_scores.append(0.8)
                flagged_evidence.append(FlaggedEvidenceItem(
                    modality=ModalityType.METADATA.value,
                    timestamp_or_region="File EXIF Header",
                    indicator="Editing/Generative Software Signature",
                    reason=indicator
                ))

        # 2. Audio Signals
        has_audio = audio_info.get("has_audio", False)
        if has_audio:
            avg_aud_fake = audio_info.get("average_voice_clone_score", 0.0)
            peak_aud_fake = audio_info.get("peak_voice_clone_score", 0.0)
            deepfake_scores.append(avg_aud_fake)

            if peak_aud_fake >= 0.65:
                flagged_evidence.append(FlaggedEvidenceItem(
                    modality=ModalityType.AUDIO.value,
                    timestamp_or_region="Audio Segment",
                    indicator="Synthetic Voice Clone (Wav2Vec2)",
                    reason=f"Wav2Vec2 classifier detected synthetic vocoder patterns with confidence {peak_aud_fake:.2f}."
                ))

        # 3. Cross-Modal Signals
        lip_sync = cross_modal_info.get("lip_sync") or {}
        caption_align = cross_modal_info.get("caption_alignment") or {}
        scene_cons = cross_modal_info.get("scene_consistency") or {}

        # Lip sync check
        lip_corr = lip_sync.get("correlation_score")
        if lip_corr is not None and lip_corr < 0.10 and lip_sync.get("faces_detected", False):
            deepfake_scores.append(0.75)
            flagged_evidence.append(FlaggedEvidenceItem(
                modality=ModalityType.CROSS_MODAL.value,
                timestamp_or_region="Visual Mouth vs Audio Energy",
                indicator="Lip-Sync Acoustic Desynchronization",
                reason=f"MediaPipe mouth opening shows poor correlation (r={lip_corr}) with Librosa acoustic energy, suggesting audio dubbing or face replacement."
            ))

        # Caption vs Content check
        cap_verdict = caption_align.get("alignment_verdict", "")
        if "Mismatch" in cap_verdict:
            flagged_evidence.append(FlaggedEvidenceItem(
                modality=ModalityType.CROSS_MODAL.value,
                timestamp_or_region="Caption vs Visual Embeddings",
                indicator="Contextual Caption Mismatch (CLIP)",
                reason=f"CLIP similarity ({caption_align.get('max_similarity', 0.0):.2f}) indicates visual content does not substantiate post caption."
            ))

        # Scene Consistency check
        if scene_cons.get("mismatch_detected", False):
            flagged_evidence.append(FlaggedEvidenceItem(
                modality=ModalityType.CROSS_MODAL.value,
                timestamp_or_region="Acoustic Background vs Visual Scene",
                indicator="Environmental Inconsistency (AST vs CLIP)",
                reason=scene_cons.get("explanation", "Discrepancy detected between background audio tags and visual setting.")
            ))

        # Compute aggregate probability
        if deepfake_scores:
            aggregated_prob = float(min(1.0, max(0.0, np.mean(deepfake_scores) * 0.5 + max(deepfake_scores) * 0.5)))
        else:
            aggregated_prob = 0.1

        # Determine verdict and confidence
        if aggregated_prob >= 0.65:
            verdict = "Manipulated / Deepfake"
            conf = ConfidenceLevel.HIGH if len(flagged_evidence) >= 2 else ConfidenceLevel.MODERATE
        elif aggregated_prob >= 0.40:
            verdict = "Suspicious / Inconclusive"
            conf = ConfidenceLevel.MODERATE
        else:
            if "Mismatch" in cap_verdict:
                verdict = "Out-of-Context Misinformation"
                conf = ConfidenceLevel.MODERATE
            else:
                verdict = "Likely Authentic"
                conf = ConfidenceLevel.HIGH if (has_audio and len(visual_info.get("frame_predictions", [])) > 5) else ConfidenceLevel.MODERATE

        # Speech transcript comparison to caption
        transcript_text = audio_info.get("full_transcript", "")
        if caption and transcript_text and transcript_text != "No audio stream detected in media file.":
            trans_vs_cap = f"Speech discusses: '{transcript_text[:100]}...' compared to caption: '{caption[:100]}...'."
        else:
            trans_vs_cap = "No contradictory speech patterns identified between transcript and caption."

        cross_modal_findings = CrossModalFindings(
            lip_sync=lip_sync.get("explanation", "Lip-sync correlation was not evaluable."),
            caption_vs_content=f"Verdict: {cap_verdict}. Max CLIP similarity: {caption_align.get('max_similarity', 0.0):.2f}.",
            audio_vs_visual=scene_cons.get("explanation", "Scene consistency evaluation completed."),
            transcript_vs_caption=trans_vs_cap
        )

        summary = (
            f"Forensic analysis of media (SHA-256: {sha256_hash[:12]}...) yielded an aggregated "
            f"manipulation score of {aggregated_prob:.2f} resulting in verdict '{verdict}'. "
            f"Found {len(flagged_evidence)} anomalous indicator(s) across visual, acoustic, and cross-modal pipelines."
        )

        limitations = [
            "Social media platforms (Instagram, YouTube) apply aggressive lossy compression and re-encoding which can degrade ELA fidelity and introduce high-frequency artifacts.",
            "Lip-sync correlation is sensitive to camera angle, rapid head turns, and background music overlays.",
            "Visual deepfake ViT models may exhibit reduced sensitivity on novel generative architectures not present in training distributions."
        ]

        human_checks = [
            "Perform reverse image search (Google Lens, Yandex, TinEye) on uncompressed keyframes to locate earliest public appearance.",
            "Inspect high-frequency facial boundaries, earlobes, and teeth under contrast enhancement for blending seams.",
            "Cross-reference speaker voice timbre against verified authentic speeches from the same individual."
        ]

        return ForensicReport(
            sha256_hash=sha256_hash,
            verdict=verdict,
            deepfake_probability=round(aggregated_prob, 3),
            confidence_level=conf,
            summary=summary,
            flagged_evidence=flagged_evidence,
            cross_modal_findings=cross_modal_findings,
            limitations_and_caveats=limitations,
            recommended_human_checks=human_checks
        )
