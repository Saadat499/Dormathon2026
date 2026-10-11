"""Contract conformance: every endpoint in docs/API_CONTRACT.md exists and returns the documented fields
with the documented types. Extra fields are allowed (they are listed in the contract's deviation table);
missing or renamed contract fields are not."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from .conftest import cxr_png, upload, wait_done

CONTRACT = Path(__file__).resolve().parents[2] / "docs" / "API_CONTRACT.md"

STUDY = {
    "study_id": str, "findings": list, "priority": (str, type(None)), "confidence_band": str, "ood_flag": bool,
    "heatmap_path": (str, type(None)), "uploaded_at": str, "wait_seconds": int, "rank": (int, type(None)),
    "effective_priority": str, "priority_reason": str, "clinician_urgent": bool, "status": str,
    "flagged_disagreement": bool, "plain_language": str,
}
PRIORITIES = {"raised", "normal", "unsure"}
REASONS = {"model", "clinician_request", "coordinator_override", "wait_cap"}
STATUSES = {"processing", "awaiting_review", "needs_review", "reviewed"}
ISO = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def check(obj: dict, spec: dict) -> None:
    for key, typ in spec.items():
        assert key in obj, f"contract field '{key}' is missing"
        assert isinstance(obj[key], typ), f"'{key}' has type {type(obj[key]).__name__}"
        if typ is int:
            assert not isinstance(obj[key], bool)


def check_study(s: dict) -> None:
    check(s, STUDY)
    assert s["priority"] in PRIORITIES | {None} and s["effective_priority"] in PRIORITIES
    assert s["priority_reason"] in REASONS and s["status"] in STATUSES and ISO.match(s["uploaded_at"])
    assert (s["priority"] is None) == (s["status"] == "processing")
    for f in s["findings"]:
        assert set(f) == {"name", "probability"} and isinstance(f["name"], str) and 0.0 <= f["probability"] <= 1.0
    probs = [f["probability"] for f in s["findings"]]
    assert probs == sorted(probs, reverse=True), "findings are ordered, highest probability first"
    if s["ood_flag"]:
        assert s["priority"] == "unsure", "ood_flag always forces priority 'unsure'"
    if s["heatmap_path"] is not None:
        assert s["heatmap_path"] == f"/studies/{s['study_id']}/heatmap"


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_worklist_and_study(seeded_client):
    c = seeded_client
    body = c.get("/studies").json()
    check(body, {"mode": str, "max_wait_minutes": int, "items": list})
    assert body["mode"] in ("model", "fcfs")
    for s in body["items"]:
        check_study(s)
        check_study(c.get(f"/studies/{s['study_id']}").json())


def test_upload_review_urgent_override_return_a_study(seeded_client):
    c = seeded_client
    r = upload(c, cxr_png(seed=4), urgent=True)
    assert r.status_code == 202
    check_study(r.json())
    assert r.json()["status"] == "processing" and r.json()["clinician_urgent"] is True
    sid = wait_done(c, r.json()["study_id"])["study_id"]
    check_study(c.get(f"/studies/{sid}").json())
    assert c.get(f"/studies/{sid}/image").headers["content-type"].startswith("image/")
    assert c.get(f"/studies/{sid}/heatmap").status_code == 404, "404 until a heatmap exists"
    u = c.post("/studies/DEMO-LOW/urgent-request", json={"note": "n", "actor": "a"})
    check_study(u.json())
    assert u.json()["clinician_urgent"] is True
    o = c.post("/studies/DEMO-LOW/override", json={"priority": "unsure", "reason": "required", "actor": "a"})
    check_study(o.json())
    for body in ({"action": "mark_reviewed", "actor": "a"},):
        check_study(c.post("/studies/DEMO-LOW/review", json=body).json())
    check_study(c.post("/studies/DEMO-HIGH/review", json={"action": "flag_disagreement", "reason": "r", "notes": "n", "human_priority": "normal", "actor": "a"}).json())


def test_queue_settings(client):
    assert client.get("/queue/settings").json() == {"mode": "model", "max_wait_minutes": 120}
    assert client.put("/queue/settings", json={"mode": "fcfs", "max_wait_minutes": 90, "actor": "a"}).json() == {"mode": "fcfs", "max_wait_minutes": 90}


def test_threshold(client):
    t = client.get("/threshold").json()
    check(t, {"value": float, "min_sensitivity_target": float})
    p = client.post("/threshold/preview", json={"value": 0.5}).json()
    check(p, {"abstention_rate": float, "sensitivity": float, "false_alarm_rate": float, "n": int, "n_positive": int})
    ok = client.put("/threshold", json={"value": round(t["value"] - 0.05, 2), "confirm": True, "actor": "a"})
    assert ok.status_code == 200
    check(ok.json(), {"value": float, "min_sensitivity_target": float})
    assert client.put("/threshold", json={"value": 0.3}).status_code == 400, "400 unless confirm is true"
    assert client.put("/threshold", json={"value": 0.99, "confirm": True}).status_code == 400, "400 below the sensitivity target"


def test_queue_race(client):
    j = client.post("/simulations/queue-race", json={"arrivals_per_hour": 20, "reads_per_hour": 18, "urgent_share": 0.15, "hours": 8, "seed": 1}).json()
    check(j, {"assumptions": dict, "fcfs": dict, "voxelsort": dict, "read_order": dict})
    check(j["fcfs"], {"urgent_wait_min": (int, float), "other_extra_wait_min": (int, float), "missed_urgent": type(None), "wrong_flags": type(None)})
    check(j["voxelsort"], {"urgent_wait_min": (int, float), "other_extra_wait_min": (int, float), "missed_urgent": int, "wrong_flags": int})
    for arm in ("fcfs", "voxelsort"):
        assert j["read_order"][arm]
        for row in j["read_order"][arm]:
            check(row, {"id": str, "wait_min": (int, float), "truly_urgent": bool})


def test_site_readiness(client):
    r = client.post("/site-readiness", files=[("files", ("a.png", cxr_png(), "image/png"))])
    assert r.status_code == 202
    check(r.json(), {"job_id": str})
    res = client.get(f"/site-readiness/{r.json()['job_id']}").json()
    assert res["status"] in ("processing", "done")
    while res["status"] != "done":
        res = client.get(f"/site-readiness/{r.json()['job_id']}").json()
    check(res, {"status": str, "drift": dict, "target": float, "verdict": str})
    check(res["drift"], {"score": (int, float, type(None)), "warning": bool, "note": str})
    assert res["verdict"] in ("ready", "use_with_caution", "not_ready", "not_enough_evidence") and "sensitivity" in res
    s = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.95, "confidence": 0.95, "prevalence": 0.2}).json()
    check(s, {"positives_needed": int, "scans_needed": int, "note": str})


def test_sensitivity_block_shape(client):
    from app.services import site_readiness as sr
    from .conftest import ctx_of
    ctx = ctx_of(client)
    res = sr.evaluate(ctx, [], "label,score\n" + "1,0.999\n" * 5)
    check(res["sensitivity"], {"point": float, "ci_low": float, "ci_high": float, "width": float, "method": str, "n_positive": int})
    assert res["sensitivity"]["method"] == "clopper-pearson"


def test_ood_check_and_audit(seeded_client):
    c = seeded_client
    check(c.post("/ood-check", files={"file": ("a.png", cxr_png(), "image/png")}).json(), {"ood_flag": bool, "reason": str, "score": (int, float, type(None))})
    c.post("/studies/DEMO-LOW/urgent-request", json={"actor": "a"})
    for a in c.get("/audit").json():
        check(a, {"id": int, "timestamp": str, "actor": str, "action": str, "study_id": (str, type(None)), "detail": str})
        assert ISO.match(a["timestamp"])
    assert c.get("/audit.csv").text.splitlines()[0] == "id,timestamp,actor,action,study_id,detail"


def test_errors_use_the_documented_shape(client):
    for r in (client.get("/studies/NOPE"), client.post("/studies/NOPE/override", json={"priority": "raised", "reason": "x"}),
              client.put("/threshold", json={"value": 0.5})):
        assert set(r.json()) == {"detail"} and isinstance(r.json()["detail"], str)


@pytest.mark.skipif(not CONTRACT.is_file(), reason="docs/ is not next to backend/ (Docker image)")
def test_every_documented_path_is_served(client):
    """Every `METHOD | /path` row in the contract's endpoint tables must be a real route."""
    text = CONTRACT.read_text(encoding="utf-8")
    documented = set(re.findall(r"^\|\s*(GET|POST|PUT)\s*\|\s*`(/[^`]*)`", text, flags=re.MULTILINE))
    assert len(documented) >= 20
    served = {(m, r.path) for r in client.app.routes for m in getattr(r, "methods", ())}
    norm = lambda p: re.sub(r"\{[^}]+\}", "{}", p)  # noqa: E731
    served_norm = {(m, norm(p)) for m, p in served}
    missing = {(m, p) for m, p in documented if (m, norm(p)) not in served_norm}
    assert not missing, f"documented but not served: {sorted(missing)}"
