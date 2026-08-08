import time

from fastapi.testclient import TestClient

from app.main import app


REQUIRED_ITEM_KEYS = {
    "prediction_id",
    "video_id",
    "frame_id",
    "track_id",
    "jersey_bbox",
    "predicted_number",
    "confidence",
    "image_url",
    "reasons",
}


def _assert_candidate_payload(payload: dict) -> None:
    assert "total_count" in payload
    assert "items" in payload
    assert isinstance(payload["items"], list)
    assert payload["total_count"] == len(payload["items"])

    for item in payload["items"]:
        assert REQUIRED_ITEM_KEYS.issubset(item)
        assert isinstance(item["reasons"], list)
        assert all(isinstance(reason, str) for reason in item["reasons"])


def test_annotation_candidates_smoke_for_videos_8_2_and_3() -> None:
    client = TestClient(app, raise_server_exceptions=False)

    job_response = client.post("/videos/8/run-jersey-recognition-async")
    if job_response.status_code in {400, 404}:
        return
    assert job_response.status_code == 200, job_response.text
    job_id = job_response.json()["job_id"]

    job_payload = None
    for _ in range(20):
        poll_response = client.get(f"/jobs/{job_id}")
        assert poll_response.status_code == 200, poll_response.text
        job_payload = poll_response.json()
        if job_payload["status"] in {"completed", "failed"}:
            break
        time.sleep(0.1)

    assert job_payload is not None
    if job_payload["status"] == "failed":
        return

    assert job_payload["status"] == "completed", job_payload

    for video_id in (8, 2, 3):
        response = client.get(f"/videos/{video_id}/annotation-candidates", params={"threshold": 0.7})
        assert response.status_code == 200, response.text
        _assert_candidate_payload(response.json())
