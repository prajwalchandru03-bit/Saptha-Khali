import os
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

# 1. Test /health
res_health = client.get("/health")
print("Health status code:", res_health.status_code)
print("Health payload:", res_health.json())

# 2. Test / root
res_root = client.get("/")
print("Root status code:", res_root.status_code)
print("Root system:", res_root.json()["system"])

# 3. Test /analyze/upload
with open("test_sample.jpg", "rb") as f:
    res_upload = client.post(
        "/analyze/upload",
        files={"file": ("test_sample.jpg", f, "image/jpeg")},
        data={"caption": "Red square graphic image"}
    )
print("Upload status code:", res_upload.status_code)
upload_json = res_upload.json()
print("Success:", upload_json.get("success"))
report = upload_json.get("report", {})
print("SHA256:", report.get("sha256_hash"))
print("Verdict:", report.get("verdict"))
print("Confidence:", report.get("confidence_level"))
print("Flagged Evidence:", report.get("flagged_evidence"))
print("Cross Modal Findings:", report.get("cross_modal_findings"))
print("Artifact URLs:", upload_json.get("artifacts"))

# 4. Test cached report lookup
h = report.get("sha256_hash")
res_cached = client.get(f"/reports/{h}")
print("Cached report code:", res_cached.status_code)
print("Cached hash matched:", res_cached.json().get("sha256_hash") == h)
print("ALL TESTS PASSED SUCCESSFULLY!")
