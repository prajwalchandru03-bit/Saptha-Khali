"""
Comprehensive test script for TruthLens Backend & Web Dashboard
"""

import json
from pathlib import Path
from PIL import Image
import numpy as np
from fastapi.testclient import TestClient

from main import app, INCIDENTS_FILE

client = TestClient(app)

print("=" * 70)
print("TESTING TRUTHLENS FULL SUITE (BACKEND + WEB DASHBOARD)")
print("=" * 70)

# 1. Test GET / serves the web app
print("\n[1] Testing GET / serves static/index.html...")
res_root = client.get("/")
assert res_root.status_code == 200
assert "text/html" in res_root.headers.get("content-type", "")
assert "TruthLens" in res_root.text
assert "Multimodal Social Media Verification" in res_root.text
print("--> PASS: GET / successfully serves static/index.html web dashboard!")

# 2. Test GET /health
print("\n[2] Testing GET /health...")
res_health = client.get("/health")
assert res_health.status_code == 200
health_data = res_health.json()
assert health_data["status"] == "healthy"
assert health_data["max_video_frames_analysis"] == 10
print(f"--> PASS: Health check healthy! Max frames capped at: {health_data['max_video_frames_analysis']}")

# 3. Test Picture Upload with ELA Heatmap, Face Crop, and 3-Way Assessment
print("\n[3] Testing POST /analyze/upload for Picture (.jpg)...")
test_dir = Path("./test_suite_workspace")
test_dir.mkdir(parents=True, exist_ok=True)

# Generate synthetic image
img_arr = np.full((300, 300, 3), 150, dtype=np.uint8)
img_arr[50:120, 50:120] = [230, 180, 140] # face-like area
img_arr[70:85, 70:80] = [20, 20, 20]     # eye
img_arr[70:85, 95:105] = [20, 20, 20]    # eye
img_arr[100:110, 75:100] = [180, 30, 30] # mouth
# add a synthetic spliced patch to trigger ELA
img_arr[180:240, 180:240] = [255, 0, 0]

pic_path = test_dir / "test_photo.jpg"
Image.fromarray(img_arr).save(pic_path, "JPEG", quality=95)

with open(pic_path, "rb") as f:
    res_upload = client.post(
        "/analyze/upload",
        files={"file": ("test_photo.jpg", f, "image/jpeg")},
        data={"caption": "Official photograph claiming to show downtown"}
    )

assert res_upload.status_code == 200, f"Upload failed: {res_upload.text}"
upload_data = res_upload.json()
report = upload_data["report"]
print(f"Verdict: {report['verdict']}")
print(f"Deepfake Alert Triggered: {report['deepfake_alert_triggered']}")
print(f"Media Origin Breakdown:")
origin = report["media_origin_assessment"]
print(f" - Deepfake Face: {origin['deepfake_face_manipulation']['indicator']} ({origin['deepfake_face_manipulation']['reason']})")
print(f" - AI Visuals: {origin['ai_generated_visuals']['indicator']} ({origin['ai_generated_visuals']['reason']})")
print(f" - Voice / Other Voice: {origin['voice_manipulation_or_other_voice']['indicator']} ({origin['voice_manipulation_or_other_voice']['reason']})")

assert "deepfake_face_manipulation" in origin
assert "ai_generated_visuals" in origin
assert "voice_manipulation_or_other_voice" in origin
assert "draft_takedown_report" in report
assert "TRUTHLENS MULTIMODAL FORENSIC INCIDENT" in report["draft_takedown_report"]
print("--> PASS: Picture uploaded and analyzed with full 3-way assessment & takedown draft!")

# 4. Test Incident Reporting: POST /incidents/report and GET /incidents
print("\n[4] Testing POST /incidents/report & GET /incidents...")
incident_payload = {
    "sha256_hash": report["sha256_hash"],
    "source_identifier": "test_photo.jpg",
    "manipulation_type": report["verdict"],
    "takedown_report_text": report["draft_takedown_report"],
    "investigator_notes": "Forensic test verification notes",
    "platform": "Instagram"
}
res_inc = client.post("/incidents/report", json=incident_payload)
assert res_inc.status_code == 200, f"Incident report failed: {res_inc.text}"
inc_record = res_inc.json()
print(f"Created Incident Ticket ID: {inc_record['ticket_id']}")
assert inc_record["ticket_id"].startswith("TL-202")

res_list = client.get("/incidents")
assert res_list.status_code == 200
incidents = res_list.json()
print(f"Total Logged Incidents: {len(incidents)}")
assert len(incidents) >= 1
assert incidents[0]["ticket_id"] == inc_record["ticket_id"]
print("--> PASS: Incident logging and retrieval working seamlessly!")

# 5. Test Instagram Login Wall / Block Handling
print("\n[5] Testing Instagram Login Wall handling (Clean HTTP 400)...")
res_url_block = client.post(
    "/analyze/url",
    json={"url": "https://www.instagram.com/reel/C_blocked_test_sample/"}
)
assert res_url_block.status_code == 400
detail_msg = res_url_block.json()["detail"]
print(f"HTTP 400 detail message: {detail_msg}")
assert "Instagram requires login" in detail_msg or "Please upload the" in detail_msg
print("--> PASS: Instagram block correctly returns clean user-friendly HTTP 400!")

print("\n" + "=" * 70)
print("ALL TESTS COMPLETED SUCCESSFULLY!")
print("=" * 70)
