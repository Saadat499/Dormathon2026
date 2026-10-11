"""Health and honesty about the model mode, CORS, error shapes, OOD check, evidence and the audit export."""
from __future__ import annotations

import csv
import io
import json
import shutil

from fastapi.testclient import TestClient

from app.config import Settings, get_settings
from app.main import create_app
from app.services import plain_language

from .conftest import REAL_ARTIFACTS, ctx_of, cxr_png, demo_bytes, settings_for, upload, wait_done


# --- health and model mode -------------------------------------------------------------------------
def test_health_reports_what_is_really_running(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["database"]["reachable"] is True and h["database"]["journal_mode"] == "wal"
    assert h["database"]["workers"] == 1
    assert h["model_mode"] == "replay" and h["model"]["runs_a_trained_model"] is False and h["model"]["requested"] == "auto"
    assert "not run on this host" in h["model"]["description"]
    assert h["saved_predictions"]["real"] is False and h["saved_predictions"]["source"] == "synthetic_test_fixture"
    assert h["storage"]["persistent"] is False
    s = client.get("/status").json()
    assert s["queue"] == {"mode": "model", "max_wait_minutes": 120} and s["threshold"]["value"] is not None
    assert client.get("/").json()["health"] == "/health"


def test_requesting_real_without_model_files_is_reported_not_faked(tmp_path, artifacts):
    with TestClient(create_app(settings_for(tmp_path, artifacts, model_mode="real"))) as c:
        h = c.get("/health").json()
        assert h["model"]["requested"] == "real" and h["model_mode"] == "replay" and h["model"]["degraded"] is True and h["degraded"] is True
        assert any("real inference is unavailable" in p for p in h["model"]["problems"])
        s = wait_done(c, upload(c, cxr_png(seed=9)).json()["study_id"])
        assert s["findings"] == [] and s["priority"] == "unsure" and s["model_source"] == "none", "no fabricated probabilities"


def test_no_artifacts_means_no_model_output_at_all(bare_client):
    c = bare_client
    h = c.get("/health").json()
    assert h["model_mode"] == "none" and h["degraded"] is True and h["evidence"]["loaded"] is False
    s = wait_done(c, upload(c).json()["study_id"])
    assert s["priority"] == "unsure" and s["findings"] == [] and "no model is installed" in s["plain_language"]
    assert c.get("/evidence").status_code == 404
    assert c.get("/studies").json()["items"][0]["study_id"] == s["study_id"], "the scan is still visible"


def test_fake_mode_is_opt_in_and_labelled_everywhere(tmp_path, artifacts):
    assert get_settings().model_mode == "auto", "fake is never the default"
    with TestClient(create_app(settings_for(tmp_path, artifacts, model_mode="fake"))) as c:
        h = c.get("/health").json()
        assert h["model_mode"] == "fake" and h["model"]["runs_a_trained_model"] is False
        assert any("fabricated" in p for p in h["model"]["problems"])
        s = wait_done(c, upload(c).json()["study_id"])
        assert s["model_source"] == "fake" and s["plain_language"].startswith("FAKE DEVELOPMENT OUTPUT")
        assert s["model_version"] == "fake-dev" and s["findings"]


def test_tampered_model_file_is_refused(tmp_path):
    """A model file that does not match Member A's MANIFEST checksum must not be loaded."""
    if not (REAL_ARTIFACTS / "head.npz").is_file():
        import pytest
        pytest.skip("model files are not installed")
    art = tmp_path / "art"
    art.mkdir()
    for name in ("MANIFEST.json", "SOURCE.json", "backbone.json", "policy.json", "head.npz", "ood.joblib", "saved_predictions_nih.csv.gz"):
        shutil.copy2(REAL_ARTIFACTS / name, art / name)
    (art / "backbone_densenet121-onnx.onnx").write_bytes(b"not the real weights")
    with TestClient(create_app(settings_for(tmp_path, art, model_mode="real"))) as c:
        h = c.get("/health").json()
        assert h["model_mode"] == "none" and h["model"]["degraded"] is True
        assert any("does not match the checksum" in p for p in h["model"]["problems"])


def test_settings_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    assert "*" not in get_settings().allowed_origins, "production never defaults to open CORS"
    monkeypatch.setenv("ALLOWED_ORIGINS", "https://voxelsort.vercel.app/, http://localhost:5173")
    monkeypatch.setenv("MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("MIN_SENSITIVITY_TARGET", "0.85")
    monkeypatch.setenv("MODEL_MODE", "replay")
    st = get_settings()
    assert st.allowed_origins == ["https://voxelsort.vercel.app", "http://localhost:5173"]
    assert st.max_upload_bytes == 7 * 1024 * 1024 and st.min_sensitivity_target == 0.85 and st.model_mode == "replay"
    assert st.database_url.startswith("sqlite:///") and st.database_url.endswith("voxelsort.db")
    import pytest
    with pytest.raises(ValueError):
        Settings(model_mode="pretend")


# --- CORS and errors -------------------------------------------------------------------------------
def test_cors_allows_only_listed_origins(tmp_path, artifacts):
    st = settings_for(tmp_path, artifacts, allowed_origins=["https://voxelsort.vercel.app"], allowed_origin_regex=r"https://voxelsort-[a-z0-9-]+\.vercel\.app")
    with TestClient(create_app(st)) as c:
        ok = c.get("/health", headers={"Origin": "https://voxelsort.vercel.app"})
        assert ok.headers.get("access-control-allow-origin") == "https://voxelsort.vercel.app"
        preview = c.get("/health", headers={"Origin": "https://voxelsort-git-main-team.vercel.app"})
        assert preview.headers.get("access-control-allow-origin") == "https://voxelsort-git-main-team.vercel.app"
        assert "access-control-allow-origin" not in c.get("/health", headers={"Origin": "https://evil.example"}).headers
        pre = c.options("/studies", headers={"Origin": "https://voxelsort.vercel.app", "Access-Control-Request-Method": "POST"})
        assert pre.status_code == 200 and "POST" in pre.headers["access-control-allow-methods"]
        err = c.get("/studies/NOPE", headers={"Origin": "https://voxelsort.vercel.app"})
        assert err.status_code == 404 and err.headers.get("access-control-allow-origin"), "errors are readable by the browser too"


def test_errors_always_have_a_detail_string(client):
    for r in (client.get("/nope"), client.post("/studies/x/override", json={"priority": 3}), client.put("/queue/settings", json={"mode": "model", "max_wait_minutes": "soon"}),
              client.post("/threshold/preview", content="{", headers={"Content-Type": "application/json"}), client.get("/audit?limit=0"),
              client.delete("/studies")):
        assert r.status_code in (400, 404, 405) and isinstance(r.json()["detail"], str) and r.json()["detail"], r.text


def test_unexpected_errors_become_a_500_with_detail_and_cors(tmp_path, artifacts, monkeypatch):
    from app.services import studies
    with TestClient(create_app(settings_for(tmp_path, artifacts, allowed_origins=["https://a.example"])), raise_server_exceptions=False) as c:
        monkeypatch.setattr(studies, "list_studies", lambda *a, **k: 1 / 0)
        r = c.get("/studies", headers={"Origin": "https://a.example"})
        assert r.status_code == 500 and "unexpected error" in r.json()["detail"]
        assert r.headers.get("access-control-allow-origin") == "https://a.example"
        assert c.get("/health").status_code == 200, "the server keeps running"


def test_request_body_cap(tmp_path, artifacts):
    from app import main
    app = create_app(settings_for(tmp_path, artifacts))
    for m in app.user_middleware:
        if m.cls is main.GuardMiddleware:
            m.kwargs["max_body_bytes"] = 50_000
    with TestClient(app) as c:
        r = c.post("/studies", files={"file": ("big.png", b"0" * 200_000, "image/png")})
        assert r.status_code == 413 and "larger than" in r.json()["detail"]
        assert c.get("/studies").json()["items"] == []


# --- OOD check -------------------------------------------------------------------------------------
def test_ood_check_shape_and_sources(client, artifacts):
    r = client.post("/ood-check", files={"file": ("x.png", demo_bytes(artifacts, "DEMO-OOD"), "image/png")})
    j = r.json()
    assert r.status_code == 200 and {"ood_flag", "reason", "score"} <= set(j)
    assert j["ood_flag"] is True and j["source"] == "replay" and "saved result" in j["method"]
    plain = client.post("/ood-check", files={"file": ("x.png", cxr_png(), "image/png")}).json()
    assert plain["ood_flag"] is False and plain["source"] == "basic" and "placeholder" in plain["method"]
    assert "does not show the image is in-distribution" in plain["reason"] and "No flag does not prove" in plain["note"]
    colour = client.post("/ood-check", files={"file": ("x.png", cxr_png(colour=True), "image/png")}).json()
    assert colour["ood_flag"] is True and "colour image" in colour["reason"]
    assert client.post("/ood-check", files={"file": ("x.png", b"nope", "image/png")}).status_code == 400
    assert client.post("/ood-check").status_code == 400
    assert client.get("/audit").json() == [], "a check is not a state change"


def test_basic_check_flag_moves_an_unmatched_upload_to_unsure_never_down(client):
    s = wait_done(client, upload(client, cxr_png(colour=True)).json()["study_id"])
    assert s["ood_flag"] is True and s["priority"] == "unsure" and "basic image check" not in s["plain_language"]
    assert s["findings"] == []


# --- evidence --------------------------------------------------------------------------------------
def test_evidence_serves_real_results_with_provenance(tmp_path, artifacts):
    shutil.copy2(REAL_ARTIFACTS / "evidence.json", artifacts / "evidence.json")
    with TestClient(create_app(settings_for(tmp_path, artifacts))) as c:
        e = c.get("/evidence").json()
    assert {"calibration", "cross_dataset", "per_finding", "shortcut_audit", "failures"} <= set(e)
    assert e["provenance"]["data"] == "real" and e["provenance"]["model_version"].startswith("partA-")
    assert {"before", "after", "ece_before", "ece_after"} <= set(e["calibration"])
    assert all({"confidence", "accuracy"} <= set(p) for p in e["calibration"]["before"] + e["calibration"]["after"])
    assert isinstance(e["cross_dataset"], list) and isinstance(e["per_finding"], list) and e["per_finding"][0]["finding"]
    assert all(f["image_url"].startswith("http://testserver/evidence/failures/") for f in e["failures"])
    b = e["extras"]["member_b_simulated_cohorts"]
    assert b["data"] == "SYNTHETIC" and "NOT model performance" in b["statement"]
    text = json.dumps({k: v for k, v in e.items() if k != "extras"}).lower()
    assert "well calibrated" not in text.replace("do not describe the model as well calibrated", "").replace("no claim is made that the probabilities are well calibrated", "")
    assert "0.4428" not in text and "0.4316" not in text, "B's synthetic calibration numbers must not appear as results"


def test_failure_images_are_served_safely(client):
    assert client.get("/evidence/failures/../../SOURCE.json").status_code == 404
    assert client.get("/evidence/failures/x.png").status_code == 404
    assert client.get("/evidence/failures/missing.jpg").status_code == 404


# --- audit -----------------------------------------------------------------------------------------
def test_audit_filters_and_csv(seeded_client):
    c = seeded_client
    c.post("/studies/DEMO-LOW/override", json={"priority": "raised", "reason": "=HYPERLINK(\"http://x\")", "actor": "@mallory"})
    c.post("/studies/DEMO-HIGH/urgent-request", json={"note": "sob", "actor": "dr c"})
    full = c.get("/audit").json()
    assert all(set(a) >= {"id", "timestamp", "actor", "action", "study_id", "detail"} for a in full)
    assert all(a["timestamp"].endswith("Z") and isinstance(a["detail"], str) for a in full)
    assert [a["action"] for a in c.get("/audit?action=override").json()] == ["override"]
    assert [a["study_id"] for a in c.get("/audit?q=demo-high").json()] == ["DEMO-HIGH"]
    assert len(c.get("/audit?limit=1").json()) == 1 and c.get("/audit?limit=1").json()[0]["id"] == full[0]["id"]
    assert c.get("/audit?limit=5000").status_code == 400
    r = c.get("/audit.csv")
    assert r.headers["content-type"].startswith("text/csv") and "attachment" in r.headers["content-disposition"]
    rows = list(csv.reader(io.StringIO(r.text)))
    assert rows[0] == ["id", "timestamp", "actor", "action", "study_id", "detail"] and len(rows) == len(full) + 1
    override_row = next(row for row in rows if row[3] == "override")
    assert override_row[2].startswith("'@") and not override_row[5].startswith("="), "spreadsheet formulas are neutralised"
    assert [row[3] for row in list(csv.reader(io.StringIO(c.get("/audit.csv?action=urgent_request").text)))[1:]] == ["urgent_request"]


def test_actor_is_optional_and_cleaned(seeded_client):
    c = seeded_client
    c.post("/studies/DEMO-LOW/urgent-request", json={})
    c.post("/studies/DEMO-HIGH/urgent-request", json={"actor": "  dr\x00 evil\n" + "x" * 150})
    log = c.get("/audit?action=urgent_request").json()
    assert log[1]["actor"] == "unnamed"
    assert c.post("/studies/DEMO-OOD/urgent-request", json={"actor": "y" * 500}).status_code == 400, "absurdly long names are refused"
    assert "\x00" not in log[0]["actor"] and "\n" not in log[0]["actor"] and len(log[0]["actor"]) <= 80


# --- plain language --------------------------------------------------------------------------------
def test_plain_language_templates_are_fixed_and_honest():
    seen = set()
    for source in ("real", "replay", "saved_demo", "fake", "none"):
        for pr in ("raised", "unsure", "normal"):
            for reason in ("model", "clinician_request", "wait_cap", "coordinator_override"):
                for ood in (False, True):
                    for findings in ([], [{"name": "effusion", "probability": 0.27}, {"name": "atelectasis", "probability": 0.5}]):
                        t = plain_language.build(processing=False, source=source, findings=findings, model_priority=pr, ood_flag=ood,
                                                 ood_reason="different distribution from the training images" if ood else None,
                                                 model_note=None, reason=reason, effective_priority=pr, max_wait_minutes=120,
                                                 urgent_findings=["effusion"])
                        assert t.endswith(plain_language.DECISION_SUPPORT)
                        assert "calibrat" not in t.lower() and "diagnos" not in t.lower().replace("not a diagnosis", "")
                        if findings and not ood:
                            assert "27% for effusion" in t, "the urgent finding that drove the score is named, with its real probability"
                        seen.add(t)
    assert len(seen) > 20
    assert "still being processed" in plain_language.build(
        processing=True, source="none", findings=[], model_priority=None, ood_flag=False, ood_reason=None, model_note=None,
        reason="model", effective_priority="unsure", max_wait_minutes=120)


def test_ctx_lanes_fall_back_to_a_policy_when_predictions_are_missing(tmp_path, artifacts):
    (artifacts / "saved_predictions_nih.csv.gz").unlink()
    with TestClient(create_app(settings_for(tmp_path, artifacts))) as c:
        ctx = ctx_of(c)
        assert ctx.threshold is None and (ctx.lanes().t_raise, ctx.lanes().t_unsure) == (0.5, 0.4)
        assert "policy.json" in c.get("/threshold").json()["cutoff_source"]
        s = wait_done(c, upload(c, demo_bytes(artifacts, "DEMO-HIGH")).json()["study_id"])
        assert s["priority"] == "raised"
