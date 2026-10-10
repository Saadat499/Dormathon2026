import json

import pytest
from conftest import cxr_png, upload


def ids(c, **q):
    return [s["study_id"] for s in c.get("/studies", params=q).json()["items"]]


def test_health_and_contract_shapes(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/studies").json()
    assert set(r) == {"mode", "max_wait_minutes", "items"}
    s = r["items"][0]
    for k in ("study_id", "findings", "priority", "confidence_band", "ood_flag", "heatmap_path", "uploaded_at", "wait_seconds",
              "rank", "effective_priority", "priority_reason", "clinician_urgent", "status", "flagged_disagreement", "plain_language"):
        assert k in s
    assert not any(k.startswith("_") for k in s)
    assert [x["rank"] for x in r["items"] if x["status"] != "reviewed"] == list(range(1, 1 + sum(1 for x in r["items"] if x["status"] != "reviewed")))


def test_demo_seed_is_labelled_synthetic(client):
    for s in client.get("/studies").json()["items"]:
        assert s["plain_language"].startswith("SYNTHETIC DEMO RECORD")


def test_upload_is_202_processing_then_unsure_without_a_model(empty_client):
    c = empty_client
    r = upload(c)
    assert r.status_code == 202 and r.json()["status"] == "processing"
    sid = r.json()["study_id"]
    s = c.get(f"/studies/{sid}").json()
    assert s["status"] != "processing"
    assert s["findings"] == [] and s["priority"] == "unsure" and s["effective_priority"] == "unsure"   # never "normal"
    assert "No model output" in s["plain_language"] or s["ood_flag"]
    assert c.get(f"/studies/{sid}/image").headers["content-type"] == "image/jpeg"
    assert c.get(f"/studies/{sid}/heatmap").status_code == 404


def test_upload_validation(empty_client):
    c = empty_client
    assert c.post("/studies", files={"file": ("x.png", b"junk", "image/png")}).status_code == 400
    assert c.post("/studies", files={"file": ("x.png", b"", "image/png")}).status_code == 400
    r = c.post("/studies", files={"file": ("x.dcm", b"\x00" * 128 + b"DICM" + b"0" * 50, "application/dicom")})
    assert r.status_code == 400 and "DICOM" in r.json()["detail"]
    assert c.get("/studies/NOPE").status_code == 404 and c.get("/studies/NOPE").json() == {"detail": "Study NOPE not found"}


def test_clinician_urgent_beats_everything(client):
    sid = upload(client, urgent=True).json()["study_id"]
    items = client.get("/studies").json()["items"]
    assert items[0]["study_id"] == sid and items[0]["priority_reason"] == "clinician_request"
    sid2 = upload(client).json()["study_id"]
    r = client.post(f"/studies/{sid2}/urgent-request", json={"note": "chest pain", "actor": "dr"}).json()
    assert r["clinician_urgent"] and ids(client)[:2] == [sid, sid2]


def test_review_flow_and_conflicts(client):
    sid = upload(client).json()["study_id"]
    r = client.post(f"/studies/{sid}/review", json={"action": "flag_disagreement", "reason": "too high", "human_priority": "normal", "notes": "n", "actor": "dr"})
    assert r.status_code == 200 and r.json()["status"] == "reviewed" and r.json()["flagged_disagreement"] and r.json()["rank"] is None
    assert client.post(f"/studies/{sid}/review", json={"action": "mark_reviewed"}).status_code == 409
    assert client.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": "x"}).status_code == 409
    assert client.post(f"/studies/{sid}/review", json={"action": "bogus"}).status_code == 400
    assert client.post(f"/studies/{sid}/review", json={"action": "mark_reviewed", "human_priority": "x"}).status_code == 400


def test_override_rules(client):
    sid = upload(client).json()["study_id"]    # no model => unsure
    assert client.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": " "}).status_code == 400
    assert client.post(f"/studies/{sid}/override", json={"priority": "high", "reason": "x"}).status_code == 400
    r = client.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": "clinic call", "actor": "coord"}).json()
    assert r["effective_priority"] == "raised" and r["priority_reason"] == "coordinator_override"
    ood = upload(client, data=cxr_png(colour=True)).json()["study_id"]
    assert client.get(f"/studies/{ood}").json()["ood_flag"]
    r = client.post(f"/studies/{ood}/override", json={"priority": "normal", "reason": "looks fine"})
    assert r.status_code == 409          # an OOD image can never be set to routine


def test_queue_settings_fcfs_and_wait_cap(client):
    assert client.get("/queue/settings").json() == {"mode": "model", "max_wait_minutes": 120}
    assert client.put("/queue/settings", json={"mode": "x", "max_wait_minutes": 120}).status_code == 400
    assert client.put("/queue/settings", json={"mode": "model", "max_wait_minutes": 5}).status_code == 400
    assert client.put("/queue/settings", json={"mode": "model", "max_wait_minutes": "abc"}).status_code == 400
    r = client.put("/queue/settings", json={"mode": "fcfs", "max_wait_minutes": 60, "actor": "lead"})
    assert r.json() == {"mode": "fcfs", "max_wait_minutes": 60}
    items = client.get("/studies").json()["items"]
    open_ = [s for s in items if s["status"] != "reviewed"]
    assert [s["uploaded_at"] for s in open_] == sorted(s["uploaded_at"] for s in open_)
    client.put("/queue/settings", json={"mode": "model", "max_wait_minutes": 10})
    capped = [s for s in client.get("/studies").json()["items"] if s["priority_reason"] == "wait_cap"]
    assert capped and client.get("/studies").json()["items"][0]["priority_reason"] == "wait_cap"


def test_filters(client):
    assert all(s["effective_priority"] == "raised" for s in client.get("/studies", params={"priority": "raised"}).json()["items"])
    assert all(s["status"] == "reviewed" for s in client.get("/studies", params={"status": "reviewed"}).json()["items"])
    assert ids(client, q="SYN-0006") == ["SYN-0006"] or len(ids(client, q="SYN-")) >= 1


def test_threshold_rules(client):
    t = client.get("/threshold").json()
    assert t["min_sensitivity_target"] == 0.9
    p = client.post("/threshold/preview", json={"value": 0.3}).json()
    assert set(("abstention_rate", "sensitivity", "false_alarm_rate", "n", "n_positive")) <= set(p)
    assert client.post("/threshold/preview", json={"value": 0.7}).json()["sensitivity"] < p["sensitivity"]   # higher cut, fewer caught
    assert client.get("/threshold").json() == t                                                              # preview changes nothing
    assert client.put("/threshold", json={"value": 0.3}).status_code == 400                                  # no confirm
    assert client.put("/threshold", json={"value": 0.3, "confirm": False}).status_code == 400
    assert client.put("/threshold", json={"value": 0.95, "confirm": True}).status_code == 400                 # below sensitivity target
    ok = client.put("/threshold", json={"value": 0.3, "confirm": True, "actor": "lead"})
    assert ok.status_code == 200 and ok.json()["value"] == 0.3
    assert client.get("/audit", params={"action": "threshold_change"}).json()[0]["actor"] == "lead"
    assert client.post("/threshold/preview", json={"value": 2}).status_code == 400


def test_threshold_change_moves_the_queue(client):
    before = {s["study_id"]: s["priority"] for s in client.get("/studies").json()["items"]}
    client.put("/threshold", json={"value": 0.2, "confirm": True})
    after = {s["study_id"]: s["priority"] for s in client.get("/studies").json()["items"]}
    assert before != after


def test_queue_race(client):
    body = {"arrivals_per_hour": 20, "reads_per_hour": 18, "urgent_share": 0.15, "hours": 8, "seed": 1}
    a, b = client.post("/simulations/queue-race", json=body).json(), client.post("/simulations/queue-race", json=body).json()
    assert a == b                                           # same seed, same result
    assert a["fcfs"]["missed_urgent"] is None and a["fcfs"]["wrong_flags"] is None and a["fcfs"]["other_extra_wait_min"] == 0
    assert a["voxelsort"]["missed_urgent"] is not None and a["voxelsort"]["wrong_flags"] is not None
    assert a["voxelsort"]["urgent_wait_min"] <= a["fcfs"]["urgent_wait_min"]
    row = a["read_order"]["voxelsort"][0]
    assert {"id", "wait_min", "truly_urgent", "urgent"} <= set(row)
    assert a["n_cases"] > 0 and "not clinical proof" in a["assumptions"]["note"] and "SYNTHETIC" in a["assumptions"]["note"]
    assert client.post("/simulations/queue-race", json={**body, "seed": 2}).json() != a
    assert client.post("/simulations/queue-race", json={**body, "hours": 500}).status_code == 400
    assert client.post("/simulations/queue-race", json={**body, "arrivals_per_hour": 200, "hours": 24}).status_code == 400
    assert client.post("/simulations/queue-race", json={"arrivals_per_hour": "x"}).status_code == 400


def test_sample_size_endpoint(client):
    r = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.95, "confidence": 0.95, "prevalence": 0.2})
    assert r.status_code == 200 and r.json()["positives_needed"] > 0
    assert client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.85}).json()["positives_needed"] is None
    assert client.get("/site-readiness/sample-size", params={"target_sensitivity": 2, "expected_sensitivity": 0.9}).status_code == 400
    assert client.get("/site-readiness/sample-size").status_code == 400


def _csv(n_pos, n_hit, n_neg=10):
    rows = ["filename,label,score"] + [f"p{i}.png,1,{0.9 if i < n_hit else 0.1}" for i in range(n_pos)] + [f"n{i}.png,0,0.1" for i in range(n_neg)]
    return "\n".join(rows).encode()


def _site(c, n_img=3, csv_bytes=None):
    files = [("files", (f"i{i}.png", cxr_png(seed=i), "image/png")) for i in range(n_img)]
    if csv_bytes is not None:
        files.append(("labels", ("labels.csv", csv_bytes, "text/csv")))
    r = c.post("/site-readiness", files=files, data={"actor": "t"})
    assert r.status_code == 202
    return c.get(f"/site-readiness/{r.json()['job_id']}").json()


def test_site_readiness_verdicts(client):
    thr = client.get("/threshold").json()["value"]
    assert 0.1 < thr < 0.9
    r = _site(client, csv_bytes=_csv(100, 99))
    assert r["status"] == "done" and r["sensitivity"]["method"] == "clopper-pearson" and r["sensitivity"]["n_positive"] == 100
    assert r["sensitivity"]["ci_low"] < r["sensitivity"]["point"] < r["sensitivity"]["ci_high"] and r["target"] == 0.9
    assert r["verdict"] in ("ready", "use_with_caution")
    assert _site(client, csv_bytes=_csv(100, 50))["verdict"] == "not_ready"
    small = _site(client, csv_bytes=_csv(10, 10))
    assert small["verdict"] == "not_enough_evidence" and small["sensitivity"]["n_positive"] == 10
    nolab = _site(client)
    assert nolab["verdict"] == "not_enough_evidence" and nolab["sensitivity"] is None and "labels" in nolab["note"].lower()
    noscore = _site(client, csv_bytes=b"label\n1\n0\n1\n")
    assert noscore["verdict"] == "not_enough_evidence" and noscore["sensitivity"] is None and "score" in noscore["note"]
    assert client.get("/site-readiness/sr-nope").status_code == 404


def test_site_readiness_bad_inputs(client):
    assert client.post("/site-readiness").status_code == 400
    r = client.post("/site-readiness", files=[("labels", ("l.csv", b"a,b\n1,2", "text/csv"))])
    assert r.status_code == 400 and "label" in r.json()["detail"]
    r = client.post("/site-readiness", files=[("labels", ("l.csv", b"label\n7", "text/csv"))])
    assert r.status_code == 400
    assert client.post("/site-readiness", files=[("files", ("a.png", b"junk", "image/png"))]).status_code == 400


def test_ood_check_endpoint(client):
    ok = client.post("/ood-check", files={"file": ("a.png", cxr_png(), "image/png")}).json()
    assert ok["ood_flag"] is False and "reason" in ok and "score" in ok and "not prove" in ok["note"]
    assert client.post("/ood-check", files={"file": ("a.png", cxr_png(colour=True), "image/png")}).json()["ood_flag"] is True
    assert client.post("/ood-check", files={"file": ("a.txt", b"hello", "text/plain")}).status_code == 400


def test_audit_log_and_csv(client):
    sid = upload(client, name='=HYPERLINK("x").png', actor="@evil").json()["study_id"]
    client.post(f"/studies/{sid}/review", json={"action": "mark_reviewed", "actor": "dr"})
    log = client.get("/audit").json()
    assert [e["action"] for e in log[:2]] == ["mark_reviewed", "upload"] and set(log[0]) == {"id", "timestamp", "actor", "action", "study_id", "detail"}
    assert client.get("/audit", params={"action": "upload"}).json()[0]["study_id"] == sid
    assert client.get("/audit", params={"q": sid.lower()}).json()
    csv_text = client.get("/audit.csv").text
    assert csv_text.splitlines()[0] == "id,timestamp,actor,action,study_id,detail"
    assert "'@evil" in csv_text and ",@evil," not in csv_text    # formula text is neutralised
    assert client.get("/audit", params={"limit": 0}).status_code == 400


def test_every_state_change_is_audited(client):
    sid = upload(client).json()["study_id"]
    client.post(f"/studies/{sid}/urgent-request", json={})
    client.post(f"/studies/{sid}/override", json={"priority": "raised", "reason": "r"})
    client.put("/queue/settings", json={"mode": "fcfs", "max_wait_minutes": 100})
    client.put("/threshold", json={"value": 0.3, "confirm": True})
    _site(client)
    client.post(f"/studies/{sid}/review", json={"action": "mark_reviewed"})
    seen = {e["action"] for e in client.get("/audit").json()}
    assert {"upload", "urgent_request", "override", "queue_settings", "threshold_change", "site_readiness_run", "mark_reviewed"} <= seen


def test_status_is_honest(client):
    s = client.get("/status").json()
    assert s["model"]["runs_a_trained_model"] is False and s["saved_predictions"]["real"] is False
    assert s["storage"]["persistent"] is False and s["evidence"]["data"] == "synthetic"


def test_evidence_is_labelled_and_never_says_well_calibrated(client):
    ev = client.get("/evidence").json()
    text = json.dumps(ev).lower()
    assert "well calibrated" not in text and "well-calibrated" not in text
    assert ev["provenance"]["data"] == "synthetic" and all("synthetic" in r["data"] for r in ev["per_finding"])
    assert ev["calibration"]["ece_after"] > 0.3 and ev["calibration"]["before"] == []
    assert all({"confidence", "accuracy"} <= set(p) for p in ev["calibration"]["after"])
    assert "cross_dataset" not in ev and ev["failures"] == []


def test_cors_allows_vercel_and_blocks_others(client):
    ok = client.get("/health", headers={"Origin": "https://voxelsort-abc.vercel.app"})
    assert ok.headers.get("access-control-allow-origin") == "https://voxelsort-abc.vercel.app"
    assert "access-control-allow-origin" not in client.get("/health", headers={"Origin": "https://evil.example"}).headers


def test_replay_mode_uses_saved_predictions_and_says_so(replay_client):
    c = replay_client
    s = c.get("/status").json()
    assert s["saved_predictions"]["real"] is True and s["model"]["mode"] == "replay" and s["model"]["runs_a_trained_model"] is False
    # seeded queue replays real saved rows
    items = c.get("/studies").json()["items"]
    assert items and all(i["plain_language"].startswith("Replayed from a saved model prediction") for i in items)
    # an upload whose name matches a saved id gets that saved result; the model is NOT run
    hit = upload(c, name="A-000.png").json()["study_id"]
    h = c.get(f"/studies/{hit}").json()
    assert h["findings"][0] == {"name": "pneumothorax", "probability": 0.9} and h["priority"] == "raised"
    assert h["plain_language"].startswith("Replayed")
    # no saved match: unsure, never normal
    miss = c.get(f"/studies/{upload(c, name='new-scan.png').json()['study_id']}").json()
    assert miss["findings"] == [] and miss["priority"] == "unsure"
    # real cohort drives the threshold and the race, and the race note no longer says synthetic
    assert c.post("/threshold/preview", json={"value": 0.5}).json()["data_source"] == "replay"
    race = c.post("/simulations/queue-race", json={"arrivals_per_hour": 20, "reads_per_hour": 18, "urgent_share": 0.15, "hours": 4, "seed": 3}).json()
    assert "SYNTHETIC" not in race["assumptions"]["note"] and "replay" in race["assumptions"]["scores"]


def test_unreadable_artifacts_fall_back_loudly(tmp_path):
    from fastapi.testclient import TestClient
    from app.main import create_app
    art = tmp_path / "a"
    art.mkdir()
    (art / "predictions.json").write_text("{not json")
    with TestClient(create_app(data_dir=tmp_path / "d", artifact_dir=art)) as c:
        s = c.get("/status").json()["saved_predictions"]
        assert s["real"] is False and s["problems"] and "predictions.json" in s["problems"][0]
