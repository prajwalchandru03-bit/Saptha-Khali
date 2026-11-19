"""
TruthLens: Multimodal Digital Forensics & Deepfake Detection Backend
FastAPI server orchestrating ingestion, visual forensics, audio forensics,
cross-modal consistency checks, and Gemini forensic correlation.
"""

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from pipeline import (
    AudioAnalyzer,
    ConfidenceLevel,
    Correlator,
    CrossModalAnalyzer,
    ForensicReport,
    IngestPipeline,
    VisualAnalyzer,
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
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./truthlens_workspace")).resolve()
TEMP_DIR = Path(os.getenv("TEMP_DIR", "./tmp")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
TEMP_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = os.getenv("DEVICE", "cpu")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
MAX_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES_ANALYSIS", "30"))

# Initialize FastAPI application
app = FastAPI(
    title="TruthLens Multimodal Forensic Backend",
    description=(
        "Forensic analysis and deepfake detection engine for social media posts "
        "(YouTube/Instagram URLs or uploaded media). Evaluates visual frames, "
        "audio voice clones, lip-sync dynamics, CLIP caption alignment, and scene consistency, "
        "synthesized via Google Gemini API into a strict forensic report."
    ),
    version="1.0.0",
)

# Configure CORS for dashboard and web extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory for inspecting ELA heatmaps, frames, and extracted audio
app.mount("/artifacts", StaticFiles(directory=str(OUTPUT_DIR)), name="artifacts")

# Initialize Pipeline Singletons
logger.info("Initializing TruthLens Forensics Pipelines...")
ingest_pipeline = IngestPipeline(output_root=OUTPUT_DIR)
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


class FullAnalysisResponse(BaseModel):
    """Complete forensic analysis output containing the strict report and metrics."""
    success: bool
    report: ForensicReport
    artifacts: ArtifactUrls
    pipeline_telemetry: Dict[str, Any]


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
    Executes the end-to-end multimodal pipeline:
    1. Ingestion: Download/upload + SHA-256 + FFmpeg 1 fps split & 16kHz audio
    2. Visual Forensics: EXIF + ELA Heatmap PNG + ViT Deepfake detection
    3. Audio Forensics: 4-sec chunking + Wav2Vec2 voice clone + Whisper transcription
    4. Cross-Modal Forensics: Lip-sync FaceMesh/RMS + CLIP caption match + Scene AST/CLIP
    5. Gemini Correlation: Synthesizes strict Pydantic JSON report
    """
    logger.info(f"Starting analysis. Source: {source} (is_url={is_url})")

    # Step 1: Ingestion
    ingest_res = ingest_pipeline.process(
        source=source,
        is_url=is_url,
        original_filename=original_filename
    )

    # Use caption from URL metadata if none was explicitly provided
    if not caption and ingest_res.metadata.get("description"):
        caption = ingest_res.metadata.get("description")

    # Step 2: Visual Forensics
    visual_res = visual_analyzer.analyze(
        frame_paths=ingest_res.frame_paths,
        frame_timestamps=ingest_res.frame_timestamps,
        work_dir=ingest_res.work_dir,
        max_frames=MAX_FRAMES
    )

    # Step 3: Audio Forensics
    audio_res = audio_analyzer.analyze(ingest_res.audio_path)

    # Step 4: Cross-Modal Forensics
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
        "metadata": ingest_res.metadata
    }

    visual_dict = {
        "exif_analysis": {
            "has_exif": visual_res.exif_analysis.has_exif,
            "software": visual_res.exif_analysis.software,
            "camera_make": visual_res.exif_analysis.camera_make,
            "camera_model": visual_res.exif_analysis.camera_model,
            "suspicious_indicators": visual_res.exif_analysis.suspicious_indicators,
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
        "is_flagged_deepfake": visual_res.is_flagged_deepfake,
        "top_flagged_frames": [
            {
                "frame_path": fp.frame_path,
                "timestamp": fp.timestamp,
                "fake_score": fp.fake_score,
                "predicted_label": fp.predicted_label
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
        "detected_language": audio_res.detected_language,
        "full_transcript": audio_res.full_transcript,
        "transcript_segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in audio_res.transcript_segments[:15]
        ],
        "top_flagged_chunks": [
            {
                "start": c.start_seconds,
                "end": c.end_seconds,
                "fake_score": c.fake_score,
                "label": c.predicted_label
            }
            for c in audio_res.top_flagged_chunks
        ]
    }

    cross_modal_dict = {
        "lip_sync": {
            "correlation_score": cross_modal_res.lip_sync.correlation_score,
            "faces_detected": cross_modal_res.lip_sync.faces_detected,
            "verdict": cross_modal_res.lip_sync.verdict,
            "explanation": cross_modal_res.lip_sync.explanation
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
            "explanation": cross_modal_res.scene_consistency.explanation
        },
        "is_cross_modal_anomaly_detected": cross_modal_res.is_cross_modal_anomaly_detected
    }

    # Pass paths of top flagged frames to Gemini correlator for direct multimodal vision
    flagged_paths = [fp.frame_path for fp in visual_res.top_flagged_frames]
    if not flagged_paths and ingest_res.frame_paths:
        flagged_paths = [ingest_res.frame_paths[0]]

    # Step 6: Invoke Gemini Correlator
    report = correlator.generate_report(
        sha256_hash=ingest_res.sha256_hash,
        ingest_info=ingest_dict,
        visual_info=visual_dict,
        audio_info=audio_dict,
        cross_modal_info=cross_modal_dict,
        caption=caption,
        flagged_frame_paths=flagged_paths
    )

    # Step 7: Persist Forensic Report JSON for evidence preservation
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

    artifacts = ArtifactUrls(
        ela_heatmap=ela_url,
        audio_wav=audio_url,
        flagged_frames=frame_urls
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

@app.get("/", tags=["Info"])
def root_info():
    """Returns TruthLens API overview and health information."""
    return {
        "system": "TruthLens Multimodal Forensic Backend",
        "version": "1.0.0",
        "status": "Operational",
        "documentation": "/docs",
        "gemini_model": GEMINI_MODEL,
        "device": DEVICE,
        "endpoints": {
            "analyze_url": "POST /analyze/url",
            "analyze_upload": "POST /analyze/upload",
            "get_report": "GET /reports/{sha256_hash}",
            "health": "GET /health"
        }
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
    Ingests an Instagram or YouTube URL via yt-dlp, performs full multimodal
    digital forensic analysis, and correlates all signals with Gemini API.
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
    except Exception as e:
        logger.error(f"Failed analyzing URL '{url}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}"
        )


@app.post("/analyze/upload", response_model=FullAnalysisResponse, tags=["Forensics"])
async def analyze_uploaded_file(
    file: UploadFile = File(..., description="Uploaded video, image, or audio file"),
    caption: Optional[str] = Form(default=None, description="Optional caption or post text to verify")
):
    """
    Uploads an image, video, or audio file, executes forensic pipeline,
    and returns a strict Pydantic Forensic Report.
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
    except Exception as e:
        logger.error(f"Failed analyzing uploaded file '{file.filename}': {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Forensic analysis failed: {str(e)}"
        )
    finally:
        # Cleanup temporary upload if copied
        if temp_save_path.exists():
            try:
                temp_save_path.unlink()
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


if __name__ == "__main__":
    import uvicorn
    host = os.getenv("APP_HOST", "0.0.0.0")
    port = int(os.getenv("APP_PORT", "8000"))
    uvicorn.run("main:app", host=host, port=port, reload=True)
