"""
TruthLens Grounded AI Forensic Assistant Module
Answers investigator inquiries strictly grounded in the verified telemetry,
frame anomalies, acoustic metrics, and cross-modal findings of a specific investigation.
Uses Gemini when available, with a deterministic grounded heuristic fallback.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from pipeline.correlator import run_with_timeout

logger = logging.getLogger("truthlens.assistant")


class AssistantResponse(BaseModel):
    """Investigation-grounded forensic response."""
    investigation_id: str
    answer: str
    grounded_evidence: List[str] = Field(default_factory=list)
    confidence: str = "High"
    source: str = "Gemini-Arbiter"  # "Gemini-Arbiter" or "Grounded-Heuristic"


def _heuristic_investigation_answer(
    investigation_id: str,
    data: Dict[str, Any],
    question: str
) -> AssistantResponse:
    """Deterministic, telemetry-grounded fallback assistant."""
    q = question.lower()
    report = data.get("report") or {}
    telemetry = data.get("pipeline_telemetry") or {}
    visual_m = telemetry.get("visual_metrics") or {}
    audio_m = telemetry.get("audio_metrics") or {}
    cross_m = telemetry.get("cross_modal_metrics") or {}

    verdict = report.get("verdict", "INCONCLUSIVE")
    prob = report.get("deepfake_probability", 0.0)
    summary = report.get("summary", "")
    origin = report.get("media_origin_assessment") or {}
    face_info = origin.get("deepfake_face_manipulation") or {}
    vis_info = origin.get("ai_generated_visuals") or {}
    voice_info = origin.get("voice_manipulation_or_other_voice") or {}
    sha256 = report.get("sha256_hash", investigation_id)

    grounded_evidence: List[str] = []

    # 1. Verdict & Authenticity
    if any(k in q for k in ["verdict", "is it real", "is it fake", "authentic", "deepfake", "result", "score"]):
        ans = (
            f"**Forensic Verdict:** `{verdict}`\n\n"
            f"- **Aggregated Manipulation Probability:** `{prob * 100:.1f}%`\n"
            f"- **Confidence Level:** `{report.get('confidence_level', 'Moderate')}`\n"
            f"- **Summary:** {summary}\n\n"
            f"**Three-Way Origin Assessment:**\n"
            f"1. Face Manipulation: **{face_info.get('indicator', 'Low')}** ({face_info.get('reason', 'N/A')})\n"
            f"2. AI Visuals: **{vis_info.get('indicator', 'Low')}** ({vis_info.get('reason', 'N/A')})\n"
            f"3. Voice Manipulation: **{voice_info.get('indicator', 'Low')}** ({voice_info.get('reason', 'N/A')})"
        )
        grounded_evidence.append(f"Verdict: {verdict} ({prob * 100:.1f}%)")
        grounded_evidence.append(f"SHA-256: {sha256}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # 2. Face / Deepfake Visuals
    if any(k in q for k in ["face", "face swap", "facial", "vit", "eyes", "teeth", "seam"]):
        peak_face = visual_m.get("peak_face_fake_score", 0.0)
        avg_face = visual_m.get("average_face_fake_score", 0.0)
        flagged_frames = visual_m.get("top_flagged_frames", [])
        ans = (
            f"**Facial Forensics for Investigation `{investigation_id[:12]}`:**\n\n"
            f"- **Assessment Level:** `{face_info.get('indicator', 'Low')}`\n"
            f"- **Peak Face Anomaly Score:** `{peak_face:.2f}`\n"
            f"- **Average Face Score:** `{avg_face:.2f}`\n"
            f"- **Reasoning:** {face_info.get('reason', 'No anomalies detected.')}\n"
            f"- **Flagged Face Frames:** {len([f for f in flagged_frames if f.get('face_detected')])} frames analyzed with detected faces."
        )
        grounded_evidence.append(f"Face Manipulation: {face_info.get('indicator', 'Low')}")
        grounded_evidence.append(f"Peak Face Score: {peak_face:.2f}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # 3. Audio / Voice Clone / Speech
    if any(k in q for k in ["audio", "voice", "clone", "speech", "wav2vec", "music", "song"]):
        has_aud = audio_m.get("has_audio", False)
        if not has_aud:
            ans = f"This media file does not contain an audio stream. Audio forensics was not applicable."
            grounded_evidence.append("No audio stream present.")
        else:
            peak_aud = audio_m.get("peak_voice_clone_score", 0.0)
            avg_aud = audio_m.get("average_voice_clone_score", 0.0)
            music = audio_m.get("background_music_detected", False)
            spikes = audio_m.get("synthetic_voice_spikes", [])
            ans = (
                f"**Acoustic & Voice Forensics:**\n\n"
                f"- **Voice Manipulation Assessment:** `{voice_info.get('indicator', 'Low')}`\n"
                f"- **Peak Voice Clone Score:** `{peak_aud:.2f}`\n"
                f"- **Average Voice Clone Score:** `{avg_aud:.2f}`\n"
                f"- **Background Music Detected:** `{'Yes (calibration applied)' if music else 'No'}`\n"
                f"- **Synthetic Voice Spikes Flagged:** `{len(spikes)}`\n"
                f"- **Reasoning:** {voice_info.get('reason', 'Natural speech dynamics.')}"
            )
            grounded_evidence.append(f"Voice Manipulation: {voice_info.get('indicator', 'Low')}")
            grounded_evidence.append(f"Peak Voice Clone Score: {peak_aud:.2f}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # 4. Lip Sync & Dubbed Voice
    if any(k in q for k in ["lip", "sync", "mouth", "dub", "dubbed"]):
        lip = cross_m.get("lip_sync") or {}
        ans = (
            f"**Lip-Sync & Dubbing Forensics:**\n\n"
            f"- **Dubbed Voice Detected:** `{'YES' if lip.get('dubbed_voice_detected') else 'NO'}`\n"
            f"- **Lip-Sync Verdict:** `{lip.get('verdict', 'Unknown')}`\n"
            f"- **Correlation Score:** `{lip.get('correlation_score')}`\n"
            f"- **Forensic Explanation:** {lip.get('explanation', 'Not evaluable.')}"
        )
        grounded_evidence.append(f"Lip-Sync: {lip.get('verdict', 'N/A')}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # 5. EXIF / Metadata / Provenance
    if any(k in q for k in ["exif", "metadata", "camera", "software", "provenance", "hash"]):
        exif = visual_m.get("exif_analysis") or {}
        prov = report.get("provenance") or {}
        ans = (
            f"**Evidentiary Metadata & Provenance:**\n\n"
            f"- **SHA-256 Hash:** `{sha256}`\n"
            f"- **Camera Make/Model:** `{exif.get('camera_make') or 'Stripped / None'}` / `{exif.get('camera_model') or 'None'}`\n"
            f"- **Editing Software:** `{exif.get('software') or 'None detected'}`\n"
            f"- **Platform Note:** {exif.get('platform_note', 'Standard platform recompression observed.')}"
        )
        grounded_evidence.append(f"SHA-256: {sha256}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # 6. ELA / Heatmap
    if any(k in q for k in ["ela", "heatmap", "splicing", "compression", "pixels"]):
        ela = visual_m.get("ela_result") or {}
        ans = (
            f"**Error Level Analysis (ELA) Findings:**\n\n"
            f"- **Tampering Suspicion Score:** `{ela.get('tampering_suspicion_score', 0.0):.2f}`\n"
            f"- **Error Variance:** `{ela.get('error_variance', 0.0):.2f}`\n"
            f"- **Mean Compression Error:** `{ela.get('mean_error', 0.0):.2f}`\n"
            f"- **Interpretation:** Higher localized variance indicates digital splicing or mismatched resaving compression rates."
        )
        grounded_evidence.append(f"ELA Suspicion Score: {ela.get('tampering_suspicion_score', 0.0):.2f}")
        return AssistantResponse(
            investigation_id=investigation_id,
            answer=ans,
            grounded_evidence=grounded_evidence,
            source="Grounded-Heuristic"
        )

    # Default overview
    ans = (
        f"**Investigation `{investigation_id[:12]}` Overview:**\n\n"
        f"- **Verdict:** `{verdict}` (Manipulation Probability: `{prob * 100:.1f}%`)\n"
        f"- **Summary:** {summary}\n\n"
        f"You can ask me specific questions about:\n"
        f"- **Face Swaps & Visual Artifacts** (e.g. 'Were any face manipulation cues detected?')\n"
        f"- **Voice Cloning & Dubbing** (e.g. 'Is the speech audio authentic or synthetic?')\n"
        f"- **Lip-Sync Accuracy** (e.g. 'Does the audio match mouth movements?')\n"
        f"- **Metadata & EXIF** (e.g. 'What camera or software tags were found?')\n"
        f"- **Error Level Analysis** (e.g. 'What did the ELA heatmap show?')"
    )
    grounded_evidence.append(f"Verdict: {verdict}")
    return AssistantResponse(
        investigation_id=investigation_id,
        answer=ans,
        grounded_evidence=grounded_evidence,
        source="Grounded-Heuristic"
    )


def investigation_assistant(
    investigation_id: str,
    investigation_data: Dict[str, Any],
    question: str,
    chat_history: Optional[List[Dict[str, str]]] = None
) -> AssistantResponse:
    """
    Answers questions about a specific investigation using Gemini with strict grounding,
    falling back to the deterministic grounded heuristic assistant if offline or unavailable.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    if not api_key:
        return _heuristic_investigation_answer(investigation_id, investigation_data, question)

    try:
        from google import genai
        client = genai.Client(api_key=api_key)

        # Build clean telemetry payload for grounding
        grounding_data = {
            "investigation_id": investigation_id,
            "report": investigation_data.get("report", {}),
            "telemetry": investigation_data.get("pipeline_telemetry", {})
        }

        system_instruction = (
            "You are the TruthLens Forensic AI Assistant. You assist human OSINT investigators by answering questions "
            f"strictly grounded in Investigation ID: {investigation_id}.\n\n"
            "STRICT GROUNDING RULES:\n"
            "1. Answer ONLY using facts, metrics, scores, and findings present in the provided forensic report and telemetry.\n"
            "2. Never hallucinate, invent timestamps, or guess evidence not present in the data.\n"
            "3. If the user asks something that cannot be determined from the telemetry, state clearly: 'That information was not captured in this forensic scan.'\n"
            "4. Be objective, concise, and professional. Quote specific scores or indicators where helpful."
        )

        history_context = ""
        if chat_history:
            for msg in chat_history[-6:]:
                role = "Investigator" if msg.get("role") == "user" else "Assistant"
                history_context += f"{role}: {msg.get('content')}\n"

        prompt = (
            f"{system_instruction}\n\n"
            f"INVESTIGATION FORENSIC DATA:\n{json.dumps(grounding_data, indent=2, default=str)}\n\n"
            f"CONVERSATION HISTORY:\n{history_context}\n"
            f"INVESTIGATOR QUESTION: {question}\n\n"
            "Provide a grounded, professional forensic response in Markdown format."
        )

        response = run_with_timeout(
            client.models.generate_content,
            kwargs={"model": model_name, "contents": prompt},
            timeout=10.0
        )

        if response and response.text:
            report = investigation_data.get("report") or {}
            grounded_evidence = [
                f"Verdict: {report.get('verdict', 'N/A')}",
                f"Score: {report.get('deepfake_probability', 0.0) * 100:.1f}%",
                f"SHA-256: {report.get('sha256_hash', investigation_id)}"
            ]
            return AssistantResponse(
                investigation_id=investigation_id,
                answer=response.text.strip(),
                grounded_evidence=grounded_evidence,
                source="Gemini-Arbiter"
            )
    except Exception as e:
        logger.warning(f"Gemini assistant failed: {e}. Falling back to grounded heuristic.", exc_info=False)

    return _heuristic_investigation_answer(investigation_id, investigation_data, question)
