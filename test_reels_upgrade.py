"""
Comprehensive test script for TruthLens Instagram Reels & Social Video Upgrades
"""

import json
import os
import numpy as np
from pathlib import Path
from PIL import Image
import soundfile as sf
from fastapi.testclient import TestClient

from main import app
from pipeline import (
    IngestPipeline,
    IngestionError,
    VisualAnalyzer,
    AudioAnalyzer,
    CrossModalAnalyzer,
    Correlator,
    ForensicReport,
    MediaOriginAssessment,
    AssessmentLevel
)

client = TestClient(app)

print("=" * 70)
print("RUNNING TRUTHLENS INSTAGRAM REELS OPTIMIZATION TESTS")
print("=" * 70)

# ------------------------------------------------------------------------------
# Test 1: Instagram-Ready Ingestion & Clean HTTP 400 Error Handling
# ------------------------------------------------------------------------------
print("\n[Test 1] Testing IngestionError & HTTP 400 Handling on Instagram Block...")
res_blocked = client.post(
    "/analyze/url",
    json={"url": "https://www.instagram.com/reel/C_blocked_login_wall_sim/"}
)
print("Status Code on Instagram Block:", res_blocked.status_code)
print("Detail:", res_blocked.json().get("detail"))
assert res_blocked.status_code == 400, f"Expected 400, got {res_blocked.status_code}"
assert "Instagram requires login" in res_blocked.json()["detail"] or "Please download the Reel" in res_blocked.json()["detail"]
print("--> PASS: Clean HTTP 400 returned instead of HTTP 500.")

# ------------------------------------------------------------------------------
# Test 2: Capped Keyframes & Separate Face vs Full Frame Detection
# ------------------------------------------------------------------------------
print("\n[Test 2] Testing Capped Keyframe Extraction, Face Cropping & Temporal Flicker...")

test_dir = Path("./test_reels_workspace")
test_dir.mkdir(parents=True, exist_ok=True)

# Generate a synthetic test image with a mock face-like center box
img_arr = np.full((320, 240, 3), 160, dtype=np.uint8)  # vertical Reel aspect ratio
# Draw a skin-tone oval/rectangle in the upper middle
img_arr[60:180, 70:170] = [210, 160, 120]  # skin tone
# Draw eyes and mouth
img_arr[90:105, 90:110] = [30, 30, 30]   # left eye
img_arr[90:105, 130:150] = [30, 30, 30]  # right eye
img_arr[140:155, 100:140] = [150, 40, 40] # mouth

sample_img_path = test_dir / "sample_reel_frame.jpg"
Image.fromarray(img_arr).save(sample_img_path, "JPEG", quality=95)

# Test Ingestion with max_keyframes=10 cap
ingest = IngestPipeline(output_root=test_dir, max_keyframes=10)
ingest_res = ingest.process(sample_img_path)
print(f"Ingested Frame Count: {len(ingest_res.frame_paths)} (Capped at <= 10)")
assert len(ingest_res.frame_paths) <= 10

# Test Visual Analyzer
visual = VisualAnalyzer(device="cpu")
vis_res = visual.analyze(ingest_res.frame_paths, ingest_res.frame_timestamps, ingest_res.work_dir, max_frames=10)

print(f"Platform Note for EXIF: {vis_res.exif_analysis.platform_note}")
assert vis_res.exif_analysis.platform_note is not None
assert "normal platform behavior" in vis_res.exif_analysis.platform_note
print("--> PASS: Missing EXIF recognized as standard Instagram H.264 platform behavior.")

print(f"Average Face Fake Score: {vis_res.average_face_fake_score}")
print(f"Average Full Frame Fake Score: {vis_res.average_full_frame_fake_score}")
print(f"Temporal Flicker Score: {vis_res.temporal_flicker_score}")
print(f"Temporal Verdict: {vis_res.temporal_consistency_verdict}")
print("--> PASS: Face-crop vs Full-frame and temporal consistency evaluated.")

# ------------------------------------------------------------------------------
# Test 3: Audio Forensics (Active Speech Isolation & Dubbed Voice Detection)
# ------------------------------------------------------------------------------
print("\n[Test 3] Testing Audio Forensics: Active Speech vs Silence & Dubbed Voice...")

# Generate a synthetic 8-second 16kHz audio: 4s speech tone + 4s silence
sr = 16000
t_tone = np.linspace(0, 4.0, int(sr * 4.0), endpoint=False)
speech_signal = (0.2 * np.sin(2 * np.pi * 300 * t_tone) + 0.1 * np.sin(2 * np.pi * 600 * t_tone)).astype(np.float32)
silence_signal = np.zeros(int(sr * 4.0), dtype=np.float32)
composite_audio = np.concatenate([speech_signal, silence_signal])

audio_test_path = test_dir / "test_audio_16k.wav"
sf.write(str(audio_test_path), composite_audio, sr)

audio_analyzer = AudioAnalyzer(device="cpu")
aud_res = audio_analyzer.analyze(str(audio_test_path))

print(f"Speech Chunks Analyzed: {aud_res.speech_chunks_analyzed}")
print(f"Silent Chunks Skipped: {aud_res.silent_chunks_skipped}")
assert aud_res.silent_chunks_skipped >= 1, "Expected at least 1 silent chunk skipped"
print("--> PASS: Silence chunks correctly filtered out to prevent false positive clone spikes.")

# Test Cross-Modal Lip Sync & Dubbed Voice Detection
cross_modal = CrossModalAnalyzer(device="cpu")
cm_res = cross_modal.analyze(
    frame_paths=ingest_res.frame_paths,
    frame_timestamps=ingest_res.frame_timestamps,
    audio_path=str(audio_test_path),
    caption="Check out this reel"
)

print(f"Lip-sync Verdict: {cm_res.lip_sync.verdict}")
print(f"Dubbed Voice Detected: {cm_res.lip_sync.dubbed_voice_detected}")
print(f"Dubbing Scenario: {cm_res.lip_sync.dubbing_scenario}")
print("--> PASS: Lip-sync and Dubbed Voice detection evaluated.")

# ------------------------------------------------------------------------------
# Test 4: Clear 3-Way Media Origin Assessment in Forensic Report
# ------------------------------------------------------------------------------
print("\n[Test 4] Testing 3-Way Media Origin Assessment in Forensic Report...")

correlator = Correlator()
report = correlator.generate_report(
    sha256_hash=ingest_res.sha256_hash,
    ingest_info={"media_type": "video", "duration_seconds": 8.0, "num_frames": len(ingest_res.frame_paths)},
    visual_info={
        "average_fake_score": vis_res.average_fake_score,
        "peak_fake_score": vis_res.peak_fake_score,
        "average_face_fake_score": vis_res.average_face_fake_score,
        "peak_face_fake_score": vis_res.peak_face_fake_score,
        "temporal_flicker_score": vis_res.temporal_flicker_score,
        "ela_result": vis_res.ela_result.__dict__ if vis_res.ela_result else None
    },
    audio_info={
        "has_audio": True,
        "average_voice_clone_score": aud_res.average_voice_clone_score,
        "peak_voice_clone_score": aud_res.peak_voice_clone_score,
        "synthetic_voice_spikes": aud_res.synthetic_voice_spikes,
        "background_music_detected": aud_res.background_music_detected,
        "full_transcript": "Testing speech transcript"
    },
    cross_modal_info={
        "lip_sync": cm_res.lip_sync.__dict__,
        "caption_alignment": cm_res.caption_alignment.__dict__,
        "scene_consistency": cm_res.scene_consistency.__dict__
    },
    caption="Testing Instagram Reel",
    flagged_frame_paths=ingest_res.frame_paths,
    face_crop_paths=vis_res.face_crop_paths
)

print("\n--- FORENSIC REPORT BREAKDOWN ---")
print("SHA256:", report.sha256_hash)
print("Verdict:", report.verdict)
print("Confidence:", report.confidence_level)
print("\nMEDIA ORIGIN ASSESSMENT (3-WAY BREAKDOWN):")
print("1. Deepfake Face Manipulation:", report.media_origin_assessment.deepfake_face_manipulation.indicator, "| Reason:", report.media_origin_assessment.deepfake_face_manipulation.reason)
print("2. AI Generated Video:", report.media_origin_assessment.ai_generated_video.indicator, "| Reason:", report.media_origin_assessment.ai_generated_video.reason)
print("3. Voice Manipulation / Other Voice:", report.media_origin_assessment.voice_manipulation_or_other_voice.indicator, "| Reason:", report.media_origin_assessment.voice_manipulation_or_other_voice.reason)

assert report.media_origin_assessment is not None
assert hasattr(report.media_origin_assessment, "deepfake_face_manipulation")
assert hasattr(report.media_origin_assessment, "ai_generated_video")
assert hasattr(report.media_origin_assessment, "voice_manipulation_or_other_voice")
assert hasattr(report, "limitations_and_caveats")
assert hasattr(report, "recommended_human_checks")
assert hasattr(report, "cross_modal_findings")

print("\nJSON Validation:")
raw_json = report.model_dump_json(indent=2)
parsed = json.loads(raw_json)
assert "media_origin_assessment" in parsed
assert "deepfake_face_manipulation" in parsed["media_origin_assessment"]
assert "ai_generated_video" in parsed["media_origin_assessment"]
assert "voice_manipulation_or_other_voice" in parsed["media_origin_assessment"]
print("--> PASS: Strict Pydantic JSON schema conforms with 3-way assessment breakdown.")

# ------------------------------------------------------------------------------
# Test 5: Full Upload Endpoint Test
# ------------------------------------------------------------------------------
print("\n[Test 5] Testing POST /analyze/upload with Artifacts...")
with open(sample_img_path, "rb") as f:
    upload_res = client.post(
        "/analyze/upload",
        files={"file": ("sample_reel_frame.jpg", f, "image/jpeg")},
        data={"caption": "Testing full reel upload"}
    )
print("Upload status:", upload_res.status_code)
assert upload_res.status_code == 200
data = upload_res.json()
assert data["success"] is True
assert "media_origin_assessment" in data["report"]
print("Artifact URLs returned:", data["artifacts"])

print("\n" + "=" * 70)
print("ALL INSTAGRAM REELS OPTIMIZATION TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
