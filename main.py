"""
TruthLens: Multimodal Digital Forensics & Deepfake Detection Backend
FastAPI server orchestrating ingestion, visual forensics, audio forensics,
cross-modal consistency checks, Gemini forensic correlation, and incident reporting.
Serves web dashboard at GET / (static/index.html).
"""

import json
import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import numpy as np

from pipeline import (
    AudioAnalyzer,
    ConfidenceLevel,
    Correlator,
    CrossModalAnalyzer,
    ForensicReport,
    IngestPipeline,
    IngestionError,
    VisualAnalyzer,
    compare_media,
    ComparisonResult,
    investigation_assistant,
    AssistantResponse,
    analyze_text_forensics,
    TextForensicsResult,
)

# Load environment configuration
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("truthlens.server")

# Paths and environment settings
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./truthlens_workspace")).resolve()
TEMP_DIR = Path(os.getenv("TEMP_DIR", "./tmp")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

INCIDENTS_FILE = OUTPUT_DIR / "reported_incidents.json"
if not INCIDENTS_FILE.exists():
    with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

DEVICE = os.getenv("DEVICE", "cpu")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
if "3.8" in GEMINI_MODEL:
    GEMINI_MODEL = "gemini-2.5-flash"
MAX_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES_ANALYSIS", "10"))

# Initialize FastAPI application
app = FastAPI(
    title="TruthLens Multimodal Forensic Backend",
    description=(
        "Forensic analysis and deepfake detection engine for social media posts "
        "(Instagram Reels, YouTube Shorts, or uploaded pictures/videos). "
        "Evaluates facial manipulation, AI-generated video cues, audio voice clones, "
        "lip-sync dynamics, and CLIP caption alignment, synthesized into a strict forensic report."
    ),
    version="1.2.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files and artifacts
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_DIR)), name="artifacts")

# Initialize Pipeline Singletons
logger.info("Initializing TruthLens Forensics Pipelines...")
ingest_pipeline = IngestPipeline(output_root=OUTPUT_DIR, max_keyframes=MAX_FRAMES)
visual_analyzer = VisualAnalyzer(device=DEVICE)
audio_analyzer = AudioAnalyzer(device=DEVICE)
cross_modal_analyzer = CrossModalAnalyzer(device=DEVICE)
correlator = Correlator(api_key=GEMINI_API_KEY, model_name=GEMINI_MODEL)
logger.info("TruthLens Pipelines Initialized Successfully.")


# ------------------------------------------------------------------------------
# Request & Response Schemas
# ------------------------------------------------------------------------------

class URLAnalysisRequest(BaseModel):
    """Request payload for social media URL forensic analysis."""
    url: str = Field(description="URL to Instagram post/reel, YouTube video/short, TikTok, etc.")
    caption: Optional[str] = Field(default=None, description="Accompanying post caption or text to verify")


class ArtifactUrls(BaseModel):
    """Web URLs to forensic artifacts generated during pipeline execution."""
    ela_heatmap: Optional[str] = None
    audio_wav: Optional[str] = None
    flagged_frames: List[str] = Field(default_factory=list)
    face_crops: List[str] = Field(default_factory=list)


class FullAnalysisResponse(BaseModel):
    """Complete forensic analysis output containing the strict report and metrics."""
    success: bool
    report: ForensicReport
    artifacts: ArtifactUrls
    pipeline_telemetry: Dict[str, Any]


class IncidentReportRequest(BaseModel):
    """Payload to log a human-confirmed deepfake incident."""
    sha256_hash: str
    source_identifier: str
    manipulation_type: str
    takedown_report_text: str
    investigator_notes: Optional[str] = None
    platform: Optional[str] = "Instagram"


class IncidentRecord(BaseModel):
    """Persisted incident report record."""
    ticket_id: str
    timestamp: str
    sha256_hash: str
    source_identifier: str
    manipulation_type: str
    takedown_report_text: str
    investigator_notes: Optional[str] = None
    platform: str
    status: str = "Logged / Ready for Takedown Submission"


class AssistantRequest(BaseModel):
    """Payload to question the grounded forensic AI assistant."""
    investigation_id: str = Field(description="SHA-256 hash or investigation identifier")
    question: str = Field(description="Investigator inquiry grounded in telemetry")
    chat_history: Optional[List[Dict[str, str]]] = Field(default_factory=list, description="Prior conversation messages")


class TextAnalysisRequest(BaseModel):
    """Payload to evaluate linguistic diversity and AI text indicators."""
    text: str = Field(description="Caption, article, or social media text to analyze")


# ------------------------------------------------------------------------------
# Pipeline Orchestration Helper
# ------------------------------------------------------------------------------

def run_multimodal_analysis(
    source: str,
    is_url: bool,
    caption: Optional[str] = None,
    original_filename: Optional[str] = None
) -> FullAnalysisResponse:
    """
    Executes the multimodal pipeline:
    1. Ingestion: Download/upload + SHA-256 + Capped keyframe split (default 10) & 16kHz audio
    2. Visual Forensics: EXIF (platform aware) + ELA Heatmap PNG + Face Crop ViT + Full Frame ViT + Temporal Flicker
    3. Audio Forensics: Active speech chunking + Wav2Vec2 clone spikes + Music check + Whisper transcription
    4. Cross-Modal Forensics: Lip-sync & Dubbed Voice check + CLIP caption match + Scene AST/CLIP
    5. Gemini Correlation: Synthesizes strict Pydantic JSON report with 3-way origin breakdown
    """
    logger.info(f"Starting analysis. Source: {source} (is_url={is_url})")

    # Step 1: Ingestion
    ingest_res = ingest_pipeline.process(
        source=source,
        is_url=is_url,
        original_filename=original_filename
    )

    is_picture = (ingest_res.media_type == "image")

    # Use caption from URL metadata if none was explicitly provided
    if not caption and ingest_res.metadata.get("description"):
        caption = ingest_res.metadata.get("description")

    # Step 2: Visual Forensics (capped keyframes, cropped face evaluation, temporal flicker, ELA)
    visual_res = visual_analyzer.analyze(
        frame_paths=ingest_res.frame_paths,
        frame_timestamps=ingest_res.frame_timestamps,
        work_dir=ingest_res.work_dir,
        max_frames=MAX_FRAMES
    )

    # Step 3: Audio Forensics (gracefully handles pictures or audio-less media)
    audio_res = audio_analyzer.analyze(ingest_res.audio_path)

    # Step 4: Cross-Modal Forensics (Lip-sync & Dubbed voice detection, CLIP caption similarity, AST scene tags)
    cross_modal_res = cross_modal_analyzer.analyze(
        frame_paths=ingest_res.frame_paths,
        frame_timestamps=ingest_res.frame_timestamps,
        audio_path=ingest_res.audio_path,
        caption=caption
    )

    # Step 5: Prepare payloads for Correlator
    ingest_dict = {
        "media_type": ingest_res.media_type,
        "duration_seconds": ingest_res.duration_seconds,
        "num_frames": len(ingest_res.frame_paths),
        "has_audio": bool(ingest_res.audio_path),
        "metadata": ingest_res.metadata,
        "original_filename": ingest_res.original_filename
    }

    visual_dict = {
        "exif_analysis": {
            "has_exif": visual_res.exif_analysis.has_exif,
            "software": visual_res.exif_analysis.software,
            "camera_make": visual_res.exif_analysis.camera_make,
            "camera_model": visual_res.exif_analysis.camera_model,
            "suspicious_indicators": visual_res.exif_analysis.suspicious_indicators,
            "platform_note": visual_res.exif_analysis.platform_note,
        },
        "ela_result": {
            "mean_error": visual_res.ela_result.mean_error if visual_res.ela_result else 0.0,
            "max_error": visual_res.ela_result.max_error if visual_res.ela_result else 0.0,
            "error_variance": visual_res.ela_result.error_variance if visual_res.ela_result else 0.0,
            "high_error_pixel_ratio": visual_res.ela_result.high_error_pixel_ratio if visual_res.ela_result else 0.0,
            "tampering_suspicion_score": visual_res.ela_result.tampering_suspicion_score if visual_res.ela_result else 0.0,
        } if visual_res.ela_result else None,
        "average_fake_score": visual_res.average_fake_score,
        "peak_fake_score": visual_res.peak_fake_score,
        "average_face_fake_score": visual_res.average_face_fake_score,
        "peak_face_fake_score": visual_res.peak_face_fake_score,
        "average_full_frame_fake_score": visual_res.average_full_frame_fake_score,
        "temporal_flicker_score": visual_res.temporal_flicker_score,
        "temporal_consistency_verdict": visual_res.temporal_consistency_verdict,
        "ai_generated_video_indicators": visual_res.ai_generated_video_indicators,
        "is_flagged_deepfake": visual_res.is_flagged_deepfake,
        "is_flagged_ai_video": visual_res.is_flagged_ai_video,
        "median_fake_score": visual_res.median_fake_score,
        "median_face_fake_score": visual_res.median_face_fake_score,
        "p75_face_fake_score": visual_res.p75_face_fake_score,
        "robust_face_fake_score": visual_res.robust_face_fake_score,
        "num_faces_detected": visual_res.num_faces_detected,
        "trimmed_mean_fake_score": visual_res.trimmed_mean_fake_score,
        "detection_screenshots": visual_res.detection_screenshots,
        "top_flagged_frames": [
            {
                "frame_path": fp.frame_path,
                "timestamp": fp.timestamp,
                "fake_score": fp.fake_score,
                "predicted_label": fp.predicted_label,
                "face_detected": fp.face_detected,
                "face_crop_path": fp.face_crop_path,
                "face_fake_score": fp.face_fake_score,
                "full_frame_fake_score": fp.full_frame_fake_score
            }
            for fp in visual_res.top_flagged_frames
        ]
    }

    audio_dict = {
        "has_audio": audio_res.has_audio,
        "duration_seconds": audio_res.duration_seconds,
        "average_voice_clone_score": audio_res.average_voice_clone_score,
        "peak_voice_clone_score": audio_res.peak_voice_clone_score,
        "is_flagged_voice_clone": audio_res.is_flagged_voice_clone,
        "speech_chunks_analyzed": audio_res.speech_chunks_analyzed,
        "silent_chunks_skipped": audio_res.silent_chunks_skipped,
        "synthetic_voice_spikes": audio_res.synthetic_voice_spikes,
        "background_music_detected": audio_res.background_music_detected,
        "music_interference_note": audio_res.music_interference_note,
        "detected_language": audio_res.detected_language,
        "full_transcript": audio_res.full_transcript,
        "transcript_segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in audio_res.transcript_segments[:15]
        ]
    }

    cross_modal_dict = {
        "lip_sync": {
            "correlation_score": cross_modal_res.lip_sync.correlation_score,
            "faces_detected": cross_modal_res.lip_sync.faces_detected,
            "num_frames_with_face": cross_modal_res.lip_sync.num_frames_with_face,
            "verdict": cross_modal_res.lip_sync.verdict,
            "explanation": cross_modal_res.lip_sync.explanation,
            "dubbed_voice_detected": cross_modal_res.lip_sync.dubbed_voice_detected,
            "dubbing_scenario": cross_modal_res.lip_sync.dubbing_scenario,
            "mismatch_ratio": cross_modal_res.lip_sync.mismatch_ratio
        },
        "caption_alignment": {
            "mean_similarity": cross_modal_res.caption_alignment.mean_similarity,
            "max_similarity": cross_modal_res.caption_alignment.max_similarity,
            "alignment_verdict": cross_modal_res.caption_alignment.alignment_verdict,
            "caption_evaluated": cross_modal_res.caption_alignment.caption_evaluated
        },
        "scene_consistency": {
            "top_visual_scenes": cross_modal_res.scene_consistency.top_visual_scenes,
            "top_audio_tags": cross_modal_res.scene_consistency.top_audio_tags,
            "mismatch_detected": cross_modal_res.scene_consistency.mismatch_detected,
            "mismatch_score": cross_modal_res.scene_consistency.mismatch_score,
            "explanation": cross_modal_res.scene_consistency.explanation,
            "music_detected": cross_modal_res.scene_consistency.music_detected
        },
        "is_cross_modal_anomaly_detected": cross_modal_res.is_cross_modal_anomaly_detected
    }

    # Pass paths of top flagged frames + cropped faces to Gemini for multimodal vision
    flagged_paths = [fp.frame_path for fp in visual_res.top_flagged_frames]
    if not flagged_paths and ingest_res.frame_paths:
        flagged_paths = [ingest_res.frame_paths[0]]

    ela_path = visual_res.ela_result.heatmap_path if visual_res.ela_result else None

    # Step 6: Invoke Gemini Correlator (or rule-based fallback)
    report = correlator.generate_report(
        sha256_hash=ingest_res.sha256_hash,
        ingest_info=ingest_dict,
        visual_info=visual_dict,
        audio_info=audio_dict,
        cross_modal_info=cross_modal_dict,
        caption=caption,
        flagged_frame_paths=flagged_paths,
        face_crop_paths=visual_res.face_crop_paths,
        ela_heatmap_path=ela_path,
        detection_screenshots=visual_res.detection_screenshots
    )

    # Step 7: Persist Forensic Report JSON
    report_file = Path(ingest_res.work_dir) / "forensic_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report.model_dump_json(indent=2))

    # Build relative artifact URLs for frontend / clients
    job_hash = ingest_res.sha256_hash
    ela_url = f"/artifacts/{job_hash}/ela_heatmap.png" if visual_res.ela_result else None
    audio_url = f"/artifacts/{job_hash}/audio_16k_mono.wav" if ingest_res.audio_path else None
    frame_urls = [
        f"/artifacts/{job_hash}/frames/{Path(fp.frame_path).name}"
        for fp in visual_res.top_flagged_frames
    ]
    face_urls = [
        f"/artifacts/{job_hash}/faces/{Path(fc).name}"
        for fc in visual_res.face_crop_paths
    ]

    artifacts = ArtifactUrls(
        ela_heatmap=ela_url,
        audio_wav=audio_url,
        flagged_frames=frame_urls,
        face_crops=face_urls
    )

    telemetry = {
        "sha256_hash": ingest_res.sha256_hash,
        "media_type": ingest_res.media_type,
        "duration_seconds": ingest_res.duration_seconds,
        "visual_metrics": visual_dict,
        "audio_metrics": audio_dict,
        "cross_modal_metrics": cross_modal_dict,
    }

    return FullAnalysisResponse(
        success=True,
        report=report,
        artifacts=artifacts,
        pipeline_telemetry=telemetry
    )


# ------------------------------------------------------------------------------
# API Endpoints
# ------------------------------------------------------------------------------

@app.get("/", tags=["Dashboard"])
def serve_dashboard():
    """Serves the complete TruthLens Web Dashboard at GET /."""
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {
        "system": "TruthLens Multimodal Forensic Backend",
        "version": "1.2.0",
        "status": "Operational",
        "documentation": "/docs"
    }


@app.get("/health", tags=["System"])
def health_check():
    """Checks system components, models, and dependencies."""
    from pipeline.ingest import resolve_ffmpeg_binary
    ffmpeg_bin = resolve_ffmpeg_binary()

    return {
        "status": "healthy",
        "device": DEVICE,
        "gemini_api_key_configured": bool(GEMINI_API_KEY),
        "gemini_model": GEMINI_MODEL,
        "max_video_frames_analysis": MAX_FRAMES,
        "ffmpeg_available": bool(ffmpeg_bin),
        "ffmpeg_path": ffmpeg_bin,
        "models": {
            "visual_detector": "dima806/deepfake_vs_real_image_detection",
            "audio_detector": "HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
            "transcription": "Whisper 'base'",
            "vision_language": "openai/clip-vit-base-patch32",
            "audio_scene": "MIT/ast-finetuned-audioset-10-10-0.4593",
            "facial_landmarks": "MediaPipe FaceMesh",
            "correlator": GEMINI_MODEL
        }
    }


@app.post("/analyze/url", response_model=FullAnalysisResponse, tags=["Forensics"])
def analyze_social_media_url(request: URLAnalysisRequest):
    """
    Ingests an Instagram Reel or YouTube video via yt-dlp, performs full multimodal
    digital forensic analysis, and correlates all signals with Gemini API.
    Catches login walls / rate limits gracefully with user-friendly HTTP 400.
    """
    url = request.url.strip()
    if not url:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="URL parameter cannot be empty."
        )

    try:
        response = run_multimodal_analysis(
            source=url,
            is_url=True,
            caption=request.caption
        )
        return response
    except IngestionError as ie:
        logger.warning(f"Ingestion rejected for URL '{url}': {ie.message}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ie.message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed analyzing URL '{url}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}"
        )


@app.post("/analyze/upload", response_model=FullAnalysisResponse, tags=["Forensics"])
async def analyze_uploaded_file(
    file: UploadFile = File(..., description="Uploaded video (.mp4, .mov, .webm), picture (.jpg, .png, .webp), or audio (.mp3, .wav)"),
    caption: Optional[str] = Form(default=None, description="Optional caption or post text to verify")
):
    """
    Uploads an image, video, or audio file, executes forensic pipeline,
    and returns a strict Pydantic Forensic Report with 3-way origin assessment.
    """
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file must have a filename."
        )

    temp_save_path = TEMP_DIR / file.filename
    try:
        with open(temp_save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        response = run_multimodal_analysis(
            source=str(temp_save_path),
            is_url=False,
            caption=caption,
            original_filename=file.filename
        )
        return response
    except IngestionError as ie:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ie.message
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed analyzing uploaded file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}"
        )
    finally:
        if temp_save_path.exists():
            try:
                temp_save_path.unlink()
            except Exception:
                pass


@app.post("/api/compare", response_model=ComparisonResult, tags=["Forensics"])
def compare_real_vs_ai(
    original_file: UploadFile = File(..., description="Original authentic reference photo or video"),
    suspect_file: UploadFile = File(..., description="Suspect photo or video to verify"),
):
    """
    Forensic side-by-side comparison between an authentic reference image/video and suspect media.
    Evaluates biometric facial structure, high-frequency sensor noise, ELA heatmaps, and AI visual cues.
    """
    if not original_file.filename or not suspect_file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Both original_file and suspect_file must be provided with valid filenames."
        )

    compare_dir = TEMP_DIR / f"compare_{uuid.uuid4().hex[:10]}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    orig_path = compare_dir / f"original_{original_file.filename}"
    susp_path = compare_dir / f"suspect_{suspect_file.filename}"

    try:
        with open(orig_path, "wb") as f_orig:
            shutil.copyfileobj(original_file.file, f_orig)
        with open(susp_path, "wb") as f_susp:
            shutil.copyfileobj(suspect_file.file, f_susp)

        result = compare_media(
            original_path=orig_path,
            suspect_path=susp_path,
            correlator_instance=correlator
        )
        return result
    except FileNotFoundError as fnfe:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(fnfe)
        )
    except Exception as e:
        logger.error(f"Error executing media comparison: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic comparison failed: {str(e)}"
        )
    finally:
        if compare_dir.exists():
            try:
                shutil.rmtree(compare_dir)
            except Exception:
                pass


@app.get("/reports/{sha256_hash}", response_model=ForensicReport, tags=["Forensics"])
def get_cached_report(sha256_hash: str):
    """
    Retrieves an existing forensic report by media SHA-256 hash.
    Ensures evidentiary chain of custody verification.
    """
    job_dir = OUTPUT_DIR / sha256_hash
    report_file = job_dir / "forensic_report.json"

    if not report_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No forensic report found for SHA-256 hash: {sha256_hash}"
        )

    try:
        with open(report_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return ForensicReport.model_validate(data)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error reading report: {str(e)}"
        )


@app.post("/incidents/report", response_model=IncidentRecord, tags=["Incidents"])
def report_deepfake_incident(request: IncidentReportRequest):
    """
    Saves a human-confirmed deepfake report to reported_incidents.json with a Ticket ID (e.g. TL-2026-001).
    """
    try:
        incidents: List[Dict[str, Any]] = []
        if INCIDENTS_FILE.exists():
            with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
                try:
                    incidents = json.load(f)
                except Exception:
                    incidents = []

        year = datetime.now().year
        ticket_seq = len(incidents) + 1
        ticket_id = f"TL-{year}-{ticket_seq:03d}"
        iso_timestamp = datetime.now(timezone.utc).isoformat()

        record = {
            "ticket_id": ticket_id,
            "timestamp": iso_timestamp,
            "sha256_hash": request.sha256_hash,
            "source_identifier": request.source_identifier,
            "manipulation_type": request.manipulation_type,
            "takedown_report_text": request.takedown_report_text,
            "investigator_notes": request.investigator_notes,
            "platform": request.platform or "Instagram",
            "status": "Logged / Ready for Takedown Submission"
        }

        incidents.insert(0, record)  # most recent first

        with open(INCIDENTS_FILE, "w", encoding="utf-8") as f:
            json.dump(incidents, f, indent=2)

        return IncidentRecord.model_validate(record)
    except Exception as e:
        logger.error(f"Failed logging incident report: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to record incident: {str(e)}"
        )


@app.get("/incidents", response_model=List[IncidentRecord], tags=["Incidents"])
def list_reported_incidents():
    """
    Returns all logged human-confirmed deepfake incident reports.
    """
    if not INCIDENTS_FILE.exists():
        return []
    try:
        with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [IncidentRecord.model_validate(item) for item in data]
    except Exception as e:
        logger.error(f"Error reading incidents: {e}")
        return []


@app.post("/api/assistant", response_model=AssistantResponse, tags=["AI Assistant"])
def ask_investigation_assistant(request: AssistantRequest):
    """
    Answers inquiries strictly grounded in an investigation's forensic telemetry.
    Uses Gemini when configured, with a deterministic grounded heuristic fallback.
    """
    sha = request.investigation_id.strip()
    job_dir = OUTPUT_DIR / sha
    rep_file = job_dir / "forensic_report.json"

    investigation_data: Dict[str, Any] = {"report": {}, "pipeline_telemetry": {}}
    if rep_file.exists():
        try:
            with open(rep_file, "r", encoding="utf-8") as f:
                rep_data = json.load(f)
            investigation_data["report"] = rep_data
        except Exception as e:
            logger.warning(f"Could not read forensic report for assistant: {e}")

    try:
        response = investigation_assistant(
            investigation_id=sha,
            investigation_data=investigation_data,
            question=request.question,
            chat_history=request.chat_history
        )
        return response
    except Exception as e:
        logger.error(f"Assistant query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assistant failed: {str(e)}"
        )


@app.post("/api/analyze/text", response_model=TextForensicsResult, tags=["Text Forensics"])
def analyze_text(request: TextAnalysisRequest):
    """
    Evaluates text for linguistic diversity, burstiness, repetition, and formulaic AI writing.
    Provides transparent probabilistic indicators with an explicit uncertainty disclaimer.
    """
    text = request.text.strip()
    if not text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text payload cannot be empty."
        )
    return analyze_text_forensics(text)


@app.get("/api/overview", tags=["Analytics"])
def get_dashboard_overview():
    """
    Calculates verified analytics aggregated from stored reports and logged incidents.
    Never fabricates stats; shows real telemetry or clean empty states.
    """
    scans = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                rep_file = d / "forensic_report.json"
                if rep_file.exists():
                    try:
                        with open(rep_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                            scans.append(data)
                    except Exception:
                        pass

    total_scans = len(scans)
    flagged_deepfakes = sum(
        1 for s in scans
        if "MANIPULATED" in str(s.get("verdict", "")).upper() or "DEEPFAKE" in str(s.get("verdict", "")).upper()
    )
    suspicious_count = sum(
        1 for s in scans
        if "SUSPICIOUS" in str(s.get("verdict", "")).upper() or "REVIEW" in str(s.get("verdict", "")).upper()
    )
    authentic_count = sum(
        1 for s in scans
        if "AUTHENTIC" in str(s.get("verdict", "")).upper() or "REAL" in str(s.get("verdict", "")).upper()
    )

    avg_score = float(np.mean([s.get("deepfake_probability", 0.0) for s in scans])) if scans else 0.0

    # Build recent scans
    recent = []
    sorted_scans = sorted(
        scans,
        key=lambda x: (x.get("provenance") or {}).get("analyzed_at", "") if isinstance(x.get("provenance"), dict) else "",
        reverse=True
    )
    for s in sorted_scans[:8]:
        prov = s.get("provenance") if isinstance(s.get("provenance"), dict) else {}
        recent.append({
            "sha256_hash": s.get("sha256_hash", ""),
            "verdict": s.get("verdict", "N/A"),
            "score": s.get("deepfake_probability", 0.0),
            "media_type": prov.get("media_type", "video"),
            "source": prov.get("source_identifier", s.get("sha256_hash", "")[:12]),
            "analyzed_at": prov.get("analyzed_at", "")
        })

    incidents_count = 0
    if INCIDENTS_FILE.exists():
        try:
            with open(INCIDENTS_FILE, "r", encoding="utf-8") as f:
                incidents_count = len(json.load(f))
        except Exception:
            incidents_count = 0

    return {
        "total_scans": total_scans,
        "flagged_deepfakes": flagged_deepfakes,
        "suspicious_count": suspicious_count,
        "authentic_count": authentic_count,
        "average_manipulation_score": round(avg_score, 3),
        "reported_incidents_count": incidents_count,
        "recent_scans": recent
    }


@app.get("/api/models/status", tags=["System"])
def get_models_status():
    """
    Returns real operational status, device placement, latency, and honest evaluation status.
    """
    return {
        "device": DEVICE,
        "models": [
            {
                "name": "dima806/deepfake_vs_real_image_detection",
                "role": "Facial & Visual Deepfake Classifier (ViT)",
                "type": "Hugging Face ViT",
                "status": "Loaded" if visual_analyzer.detector._is_loaded else "Ready (Lazy Load)",
                "device": DEVICE,
                "benchmark_accuracy": "Evaluation Pending",
                "calibration": "Median & Trimmed-Mean Aggregation"
            },
            {
                "name": "HyperMoon/wav2vec2-base-960h-finetuned-deepfake",
                "role": "Synthetic Speech & Voice Clone Classifier",
                "type": "Hugging Face Wav2Vec2",
                "status": "Loaded" if audio_analyzer.detector._is_loaded else "Ready (Lazy Load)",
                "device": DEVICE,
                "benchmark_accuracy": "Evaluation Pending",
                "calibration": "Speech-RMS Isolation & Music Dampening"
            },
            {
                "name": "openai/clip-vit-base-patch32",
                "role": "Visual-Language Context & Caption Consistency",
                "type": "Hugging Face CLIP",
                "status": "Loaded" if cross_modal_analyzer.clip_mgr._is_loaded else "Ready (Lazy Load)",
                "device": DEVICE,
                "benchmark_accuracy": "Evaluation Pending",
                "calibration": "Cosine Embedding Similarity"
            },
            {
                "name": "MIT/ast-finetuned-audioset-10-10-0.4593",
                "role": "Acoustic Scene & Environmental Audio Classifier",
                "type": "Hugging Face AST",
                "status": "Loaded" if cross_modal_analyzer.ast_classifier._is_loaded else "Ready (Lazy Load)",
                "device": DEVICE,
                "benchmark_accuracy": "Evaluation Pending",
                "calibration": "Spectral Scene Classification"
            },
            {
                "name": "MediaPipe FaceMesh / OpenCV",
                "role": "Facial Landmark & Lip-Sync Dynamics Tracker",
                "type": "Vision Pipeline",
                "status": "Operational",
                "device": "CPU",
                "benchmark_accuracy": "Deterministic Geometric",
                "calibration": "Square Aspect-Ratio Padding"
            },
            {
                "name": GEMINI_MODEL,
                "role": "Multimodal Forensic Arbiter (60% Fusion Weight)",
                "type": "Google GenAI Multimodal API",
                "status": "Connected" if GEMINI_API_KEY else "Fallback (Offline Heuristic)",
                "device": "Cloud API",
                "benchmark_accuracy": "Evaluation Pending",
                "calibration": "Structured JSON Schema"
            }
        ]
    }


@app.get("/api/history", tags=["Forensics"])
def get_investigation_history():
    """
    Lists all stored forensic investigations from the workspace.
    """
    history = []
    if OUTPUT_DIR.exists():
        for d in OUTPUT_DIR.iterdir():
            if d.is_dir():
                rep_file = d / "forensic_report.json"
                if rep_file.exists():
                    try:
                        with open(rep_file, "r", encoding="utf-8") as f:
                            rep = json.load(f)
                            prov = rep.get("provenance") or {}
                            history.append({
                                "sha256_hash": rep.get("sha256_hash", d.name),
                                "verdict": rep.get("verdict", "Unknown"),
                                "score": rep.get("deepfake_probability", 0.0),
                                "confidence": rep.get("confidence_level", "Moderate"),
                                "media_type": prov.get("media_type", "video"),
                                "source_identifier": prov.get("source_identifier", d.name[:12]),
                                "analyzed_at": prov.get("analyzed_at", "")
                            })
                    except Exception:
                        pass
    history.sort(key=lambda x: x.get("analyzed_at", ""), reverse=True)
    return history


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
