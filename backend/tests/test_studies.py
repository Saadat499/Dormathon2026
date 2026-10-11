"""Upload, background scoring, the worklist, review, urgent requests, overrides and their audit entries."""
from __future__ import annotations

from datetime import timedelta

from app.db import now_utc
from app.models import Study
from app.services import studies as studies_service

from .conftest import ctx_of, cxr_png, demo_bytes, upload, wait_done

STUDY_KEYS = {"study_id", "findings", "priority", "confidence_band", "ood_flag", "heatmap_path", "uploaded_at", "wait_seconds",
              "rank", "effective_priority", "priority_reason", "clinician_urgent", "status", "flagged_disagreement", "plain_language"}


def age(c, study_id: str, minutes: float) -> None:
    ctx = ctx_of(c)
    with ctx.db.session() as s:
        s.get(Study, study_id).uploaded_at = now_utc() - timedelta(minutes=minutes)


def actions(c, study_id=None):
    return [a["action"] for a in c.get("/audit").json() if study_id is None or a["study_id"] == study_id]


# --- upload and processing -----------------------------------------------------------------------
def test_upload_returns_202_processing_then_completes(client):
    r = upload(client)
    assert r.status_code == 202
    s = r.json()
    assert STUDY_KEYS <= set(s)
    assert s["status"] == "processing" and s["priority"] is None and s["findings"] == []
    assert s["effective_priority"] == "unsure", "a scan that is not scored yet must never be low risk"
    assert s["confidence_band"] == "pending" and s["heatmap_path"] is None and s["rank"] == 1
    done = wait_done(client, s["study_id"])
    assert done["status"] in ("needs_review", "awaiting_review")
    assert client.get(f"/studies/{s['study_id']}/image").status_code == 200
    assert client.get(f"/studies/{s['study_id']}/image").headers["content-type"] == "image/jpeg"
    assert client.get(f"/studies/{s['study_id']}/heatmap").status_code == 404
    assert actions(client, s["study_id"]) == ["scoring", "upload"]


def test_unmatched_upload_gets_no_model_output_and_is_unsure_not_normal(client):
    s = wait_done(client, upload(client, cxr_png(seed=55)).json()["study_id"])
    assert s["findings"] == [] and s["composite_score"] is None
    assert s["priority"] == "unsure" and s["effective_priority"] == "unsure" and s["status"] == "needs_review"
    assert s["model_source"] == "none" and s["confidence_band"] == "none"
    assert "No model output is available" in s["plain_language"]
    assert "Decision support only: a clinician reads every scan." in s["plain_language"]


def test_replay_matches_by_image_hash_only(client, artifacts):
    hit = wait_done(client, upload(client, demo_bytes(artifacts, "DEMO-HIGH"), name="anything.png").json()["study_id"])
    assert hit["model_source"] == "replay" and hit["priority"] == "raised"
    assert hit["findings"][0] == {"name": "pneumothorax", "probability": 0.91}
    assert hit["plain_language"].startswith("Replayed from a saved model output")
    # same FILE NAME as a saved study but different bytes: must not be matched
    miss = wait_done(client, upload(client, cxr_png(seed=77), name="DEMO-HIGH.png").json()["study_id"])
    assert miss["model_source"] == "none" and miss["findings"] == [] and miss["priority"] == "unsure"


def test_replayed_ood_image_is_unsure_with_no_findings(client, artifacts):
    s = wait_done(client, upload(client, demo_bytes(artifacts, "DEMO-OOD")).json()["study_id"])
    assert s["ood_flag"] is True and s["priority"] == "unsure" and s["effective_priority"] == "unsure"
    assert s["findings"] == [] and s["confidence_band"] == "none" and s["ood_reason"]
    assert "looks different from the scans VoxelSort was built on" in s["plain_language"]


def test_low_score_is_normal_and_says_it_is_not_an_all_clear(client, artifacts):
    s = wait_done(client, upload(client, demo_bytes(artifacts, "DEMO-LOW")).json()["study_id"])
    assert s["priority"] == "normal" and s["status"] == "awaiting_review"
    assert "not an all-clear" in s["plain_language"]


def test_invalid_uploads_are_rejected(client):
    def post(data, name="x.png", ctype="image/png"):
        return client.post("/studies", files={"file": (name, data, ctype)})
    assert post(b"").status_code == 400
    assert post(b"this is not an image").status_code == 400
    corrupt = cxr_png()[:200]
    r = post(corrupt)
    assert r.status_code == 400 and "could not be read" in r.json()["detail"]
    assert post(b"\x00" * 128 + b"DICM" + b"\x00" * 64, "scan.dcm", "application/dicom").status_code == 400
    assert "DICOM" in post(b"\x00" * 128 + b"DICM" + b"\x00" * 64, "scan.dcm", "application/dicom").json()["detail"]
    assert post(b"%PDF-1.4 ...", "x.pdf", "application/pdf").status_code == 400
    gif = b"GIF89a" + b"\x00" * 50
    assert post(gif, "x.gif", "image/gif").status_code == 400
    assert client.post("/studies").status_code == 400                                   # no file at all
    assert client.get("/studies").json()["items"] == [], "a rejected upload must not create a study"
    assert actions(client) == []


def test_oversized_upload_is_rejected(tmp_path, artifacts):
    from fastapi.testclient import TestClient

    from app.main import create_app

    from .conftest import settings_for
    with TestClient(create_app(settings_for(tmp_path, artifacts, max_upload_mb=1))) as c:
        big = b"\x89PNG\r\n\x1a\n" + b"0" * (1024 * 1024 + 10)
        r = c.post("/studies", files={"file": ("big.png", big, "image/png")})
        assert r.status_code == 413 and "larger than 1 MB" in r.json()["detail"]
        assert c.get("/studies").json()["items"] == []


def test_background_failure_leaves_the_scan_visible_and_unsure(client, monkeypatch):
    ctx = ctx_of(client)

    def boom(*a, **k):
        raise RuntimeError("model exploded")
    monkeypatch.setattr(ctx.model, "predict", boom)
    s = wait_done(client, upload(client).json()["study_id"])
    assert s["status"] == "needs_review" and s["priority"] == "unsure" and s["findings"] == []
    assert "processing failed" in s["plain_language"]
    log = [a for a in client.get("/audit").json() if a["study_id"] == s["study_id"] and a["action"] == "scoring"]
    assert log and "Processing failed" in log[0]["detail"] and log[0]["actor"] == "system"
    assert [x["study_id"] for x in client.get("/studies").json()["items"]] == [s["study_id"]]


def test_interrupted_processing_is_released_on_restart(client):
    ctx = ctx_of(client)
    sid = upload(client).json()["study_id"]
    with ctx.db.session() as s:
        s.get(Study, sid).status = "processing"
    assert studies_service.recover_interrupted(ctx) == 1
    s = client.get(f"/studies/{sid}").json()
    assert s["status"] == "needs_review" and "interrupted by a restart" in s["plain_language"]


# --- worklist ------------------------------------------------------------------------------------
def test_worklist_shape_filters_and_order(seeded_client):
    c = seeded_client
    body = c.get("/studies").json()
    assert set(body) == {"mode", "max_wait_minutes", "items"} and body["mode"] == "model" and body["max_wait_minutes"] == 120
    items = body["items"]
    assert [i["study_id"] for i in items] == ["DEMO-HIGH", "DEMO-OOD", "DEMO-LOW"]      # raised, unsure (OOD), normal
    assert [i["rank"] for i in items] == [1, 2, 3]
    assert all(STUDY_KEYS <= set(i) for i in items)
    assert all(i["plain_language"].startswith("Demo record") for i in items)
    assert [i["study_id"] for i in c.get("/studies?priority=unsure").json()["items"]] == ["DEMO-OOD"]
    assert c.get("/studies?priority=unsure").json()["items"][0]["rank"] == 2, "rank is the position in the whole queue"
    assert [i["study_id"] for i in c.get("/studies?status=needs_review").json()["items"]] == ["DEMO-OOD"]
    assert [i["study_id"] for i in c.get("/studies?q=demo-l").json()["items"]] == ["DEMO-LOW"]
    assert c.get("/studies?priority=urgent").status_code == 400
    assert c.get("/studies?status=done").status_code == 400
    assert c.get("/studies/NOPE").status_code == 404 and "detail" in c.get("/studies/NOPE").json()
    assert c.get("/studies/NOPE/image").status_code == 404 and c.get("/studies/NOPE/heatmap").status_code == 404
    assert c.get("/studies/DEMO-HIGH/image").headers["content-type"] == "image/png"


def test_seeding_is_idempotent(tmp_path, artifacts):
    from fastapi.testclient import TestClient

    from app.main import create_app

    from .conftest import settings_for
    st = settings_for(tmp_path, artifacts, seed_demo=True)
    with TestClient(create_app(st)) as c:
        first = [i["study_id"] for i in c.get("/studies").json()["items"]]
        sid = upload(c).json()["study_id"]
    with TestClient(create_app(st)) as c:                                              # "restart" on the same database
        again = [i["study_id"] for i in c.get("/studies").json()["items"]]
    assert len(first) == 3 and sorted(again) == sorted(first + [sid])


def test_clinician_request_beats_everything(seeded_client):
    c = seeded_client
    r = c.post("/studies/DEMO-LOW/urgent-request", json={"note": "chest pain", "actor": "dr a"})
    assert r.status_code == 200 and r.json()["clinician_urgent"] is True and r.json()["priority_reason"] == "clinician_request"
    assert r.json()["rank"] == 1 and r.json()["priority"] == "normal", "the model's own priority is not rewritten"
    assert [i["study_id"] for i in c.get("/studies").json()["items"]][0] == "DEMO-LOW"
    assert "A clinician asked for an urgent read" in r.json()["plain_language"]
    assert actions(c, "DEMO-LOW") == ["urgent_request"]
    c.post("/studies/DEMO-LOW/urgent-request", json={})                               # asking twice changes nothing
    assert actions(c, "DEMO-LOW") == ["urgent_request"]
    urgent_upload = upload(c, urgent=True).json()
    assert urgent_upload["clinician_urgent"] is True and urgent_upload["priority_reason"] == "clinician_request"
    top_two = [i["study_id"] for i in c.get("/studies").json()["items"][:2]]
    assert sorted(top_two) == sorted(["DEMO-LOW", urgent_upload["study_id"]]), "both clinician requests are ahead of every other scan"
    # inside the clinician group the rest of the rule still applies: unsure (not yet scored) before normal
    assert urgent_upload["rank"] == 1 and urgent_upload["effective_priority"] == "unsure"


def test_wait_cap_promotes_and_fcfs_switches_instantly(seeded_client):
    c = seeded_client
    age(c, "DEMO-LOW", 500)
    items = c.get("/studies").json()["items"]
    assert items[0]["study_id"] == "DEMO-LOW" and items[0]["priority_reason"] == "wait_cap" and items[0]["effective_priority"] == "normal"
    assert "waited longer than the 120-minute limit" in items[0]["plain_language"]
    r = c.put("/queue/settings", json={"mode": "model", "max_wait_minutes": 600, "actor": "lead"})
    assert r.json() == {"mode": "model", "max_wait_minutes": 600}
    assert c.get("/studies").json()["items"][0]["study_id"] == "DEMO-HIGH"              # cap raised: no promotion any more
    assert c.put("/queue/settings", json={"mode": "fcfs", "max_wait_minutes": 600}).json()["mode"] == "fcfs"
    fc = c.get("/studies").json()
    assert fc["mode"] == "fcfs"
    assert [i["study_id"] for i in fc["items"]] == ["DEMO-LOW", "DEMO-OOD", "DEMO-HIGH"] or \
        [i["uploaded_at"] for i in fc["items"]] == sorted(i["uploaded_at"] for i in fc["items"])
    assert all(i["priority_reason"] != "wait_cap" for i in fc["items"])
    c.post("/studies/DEMO-HIGH/urgent-request", json={})
    assert c.get("/studies").json()["items"][0]["study_id"] == "DEMO-HIGH", "a clinician request still counts in fcfs mode"
    assert c.get("/queue/settings").json() == {"mode": "fcfs", "max_wait_minutes": 600}
    assert actions(c).count("queue_settings") == 2
    for bad in ({"mode": "random", "max_wait_minutes": 60}, {"mode": "model", "max_wait_minutes": 5}, {"mode": "model", "max_wait_minutes": 5000}, {"mode": "model"}):
        assert c.put("/queue/settings", json=bad).status_code == 400


# --- overrides -----------------------------------------------------------------------------------
def test_override_needs_a_reason_and_is_logged(seeded_client):
    c = seeded_client
    assert c.post("/studies/DEMO-LOW/override", json={"priority": "raised"}).status_code == 400
    assert c.post("/studies/DEMO-LOW/override", json={"priority": "raised", "reason": "   "}).status_code == 400
    assert c.post("/studies/DEMO-LOW/override", json={"priority": "urgent", "reason": "x"}).status_code == 400
    assert c.post("/studies/NOPE/override", json={"priority": "raised", "reason": "x"}).status_code == 404
    assert actions(c) == ["upload"], "failed overrides write nothing (only the seed entry exists)"
    r = c.post("/studies/DEMO-LOW/override", json={"priority": "raised", "reason": "known pneumothorax on prior", "actor": "coordinator"})
    s = r.json()
    assert r.status_code == 200 and s["effective_priority"] == "raised" and s["priority"] == "normal" and s["priority_reason"] == "coordinator_override"
    assert "A coordinator set its queue priority to urgent review by hand" in s["plain_language"]
    entry = [a for a in c.get("/audit").json() if a["action"] == "override"][0]
    assert entry["actor"] == "coordinator" and "known pneumothorax on prior" in entry["detail"] and entry["study_id"] == "DEMO-LOW"
    assert entry["data"]["previous"] == "normal" and entry["data"]["priority"] == "raised"
    # a human may lower a scored, in-distribution scan
    low = c.post("/studies/DEMO-HIGH/override", json={"priority": "normal", "reason": "old film, already reported"}).json()
    assert low["effective_priority"] == "normal" and low["priority"] == "raised"
    assert [i["study_id"] for i in c.get("/studies").json()["items"]] == ["DEMO-LOW", "DEMO-OOD", "DEMO-HIGH"]


def test_ood_scan_cannot_be_overridden_to_normal(seeded_client):
    c = seeded_client
    r = c.post("/studies/DEMO-OOD/override", json={"priority": "normal", "reason": "looks fine"})
    assert r.status_code == 409 and "cannot be set to routine" in r.json()["detail"]
    assert c.get("/studies/DEMO-OOD").json()["effective_priority"] == "unsure"
    up = c.post("/studies/DEMO-OOD/override", json={"priority": "raised", "reason": "clinical concern"}).json()
    assert up["effective_priority"] == "raised" and up["priority"] == "unsure" and up["ood_flag"] is True


def test_precedence_clinician_then_cap_then_override(seeded_client):
    c = seeded_client
    c.post("/studies/DEMO-HIGH/override", json={"priority": "normal", "reason": "r"})   # human lowers the raised scan
    assert c.get("/studies/DEMO-HIGH").json()["priority_reason"] == "coordinator_override"
    age(c, "DEMO-HIGH", 300)
    s = c.get("/studies/DEMO-HIGH").json()
    assert s["priority_reason"] == "wait_cap" and s["rank"] == 1, "the wait cap promotes even a scan a human lowered"
    c.post("/studies/DEMO-OOD/urgent-request", json={})
    items = c.get("/studies").json()["items"]
    assert [(i["study_id"], i["priority_reason"]) for i in items] == [("DEMO-OOD", "clinician_request"), ("DEMO-HIGH", "wait_cap"), ("DEMO-LOW", "model")]


# --- review --------------------------------------------------------------------------------------
def test_review_and_disagreement(seeded_client):
    c = seeded_client
    assert c.post("/studies/DEMO-HIGH/review", json={"action": "approve"}).status_code == 400
    assert c.post("/studies/DEMO-HIGH/review", json={"action": "flag_disagreement", "human_priority": "low"}).status_code == 400
    assert c.post("/studies/NOPE/review", json={"action": "mark_reviewed"}).status_code == 404
    r = c.post("/studies/DEMO-HIGH/review", json={"action": "mark_reviewed", "actor": "dr b"})
    s = r.json()
    assert r.status_code == 200 and s["status"] == "reviewed" and s["rank"] is None and s["flagged_disagreement"] is False
    wait = s["wait_seconds"]
    assert c.get("/studies/DEMO-HIGH").json()["wait_seconds"] == wait, "the wait stops when the scan is read"
    items = c.get("/studies").json()["items"]
    assert items[-1]["study_id"] == "DEMO-HIGH" and [i["rank"] for i in items] == [1, 2, None], "reviewed scans stay listed"
    for path, body in (("review", {"action": "mark_reviewed"}), ("override", {"priority": "raised", "reason": "x"}), ("urgent-request", {})):
        assert c.post(f"/studies/DEMO-HIGH/{path}", json=body).status_code == 409
    f = c.post("/studies/DEMO-LOW/review", json={"action": "flag_disagreement", "reason": "Priority seems too low",
                                                 "human_priority": "raised", "notes": "subtle apical line", "actor": "dr b"}).json()
    assert f["flagged_disagreement"] is True and f["status"] == "reviewed"
    entry = [a for a in c.get("/audit").json() if a["action"] == "flag_disagreement"][0]
    assert "Priority seems too low" in entry["detail"] and "raised" in entry["detail"] and "subtle apical line" in entry["detail"]
    assert entry["data"]["model_priority"] == "normal"
    assert actions(c, "DEMO-HIGH") == ["mark_reviewed"]


def test_cannot_review_while_processing(client):
    ctx = ctx_of(client)
    sid = upload(client).json()["study_id"]
    with ctx.db.session() as s:
        s.get(Study, sid).status = "processing"
    assert client.post(f"/studies/{sid}/review", json={"action": "mark_reviewed"}).status_code == 409


def test_every_mutation_writes_exactly_one_audit_entry(seeded_client):
    c = seeded_client
    before = len(c.get("/audit").json())
    sid = wait_done(c, upload(c, actor="a").json()["study_id"])["study_id"]            # upload + scoring
    c.post(f"/studies/{sid}/urgent-request", json={"actor": "a"})
    c.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": "r", "actor": "a"})
    c.post(f"/studies/{sid}/review", json={"action": "flag_disagreement", "actor": "a"})
    c.post("/studies/DEMO-LOW/review", json={"action": "mark_reviewed", "actor": "a"})
    c.put("/queue/settings", json={"mode": "fcfs", "max_wait_minutes": 90, "actor": "a"})
    log = c.get("/audit").json()
    assert [a["action"] for a in log[:len(log) - before]] == ["queue_settings", "mark_reviewed", "flag_disagreement", "override", "urgent_request", "scoring", "upload"]
    assert [a["id"] for a in log] == sorted((a["id"] for a in log), reverse=True), "newest first"
    # reads and rejected requests write nothing
    n = len(log)
    c.get("/studies"); c.get(f"/studies/{sid}"); c.get("/queue/settings")
    c.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": "again"})   # 409: already reviewed
    assert len(c.get("/audit").json()) == n
