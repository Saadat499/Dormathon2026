"""Site Readiness: the statistics, the verdict rules and the job endpoints."""
from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import binomtest

from app.evalkit.protocol import calculate_required_sample_size
from app.services import site_readiness as sr

from .conftest import ctx_of, cxr_png, demo_bytes


# --- statistics ----------------------------------------------------------------------------------
@pytest.mark.parametrize("x,n", [(0, 10), (10, 10), (45, 50), (27, 30), (1, 1), (95, 100), (3, 40)])
def test_clopper_pearson_matches_scipy(x, n):
    lo, hi = sr.clopper_pearson(x, n, 0.95)
    ref = binomtest(x, n).proportion_ci(confidence_level=0.95, method="exact")
    assert abs(lo - ref.low) < 1e-9 and abs(hi - ref.high) < 1e-9


def test_sample_size_uses_b_formula_and_the_exact_bound(client):
    r = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.95})
    j = r.json()
    assert r.status_code == 200 and {"positives_needed", "scans_needed", "note"} <= set(j)
    b = calculate_required_sample_size(target_sensitivity=0.95, margin_of_error=0.05, prevalence=0.2, confidence_level=0.95)
    assert j["normal_approximation_positives"] == b["min_positive_cases_required"]
    n = j["exact_positives"]
    assert sr.clopper_pearson(round(0.95 * n), n)[0] >= 0.9 and sr.clopper_pearson(round(0.95 * (n - 1)), n - 1)[0] < 0.9
    assert j["positives_needed"] == max(j["normal_approximation_positives"], n)
    assert j["scans_needed"] == int(np.ceil(j["positives_needed"] / 0.2))
    more = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.92}).json()
    rare = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.95, "prevalence": 0.05}).json()
    assert more["positives_needed"] > j["positives_needed"] and rare["scans_needed"] > j["scans_needed"]


def test_sample_size_edge_cases(client):
    none = client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.9}).json()
    assert none["positives_needed"] is None and none["scans_needed"] is None and "must be higher" in none["note"]
    for params in ({"target_sensitivity": 1.2, "expected_sensitivity": 0.95}, {"target_sensitivity": 0.9, "expected_sensitivity": 0},
                   {"target_sensitivity": 0.9, "expected_sensitivity": 0.95, "confidence": 0.2},
                   {"target_sensitivity": 0.9, "expected_sensitivity": 0.95, "prevalence": 0}, {"target_sensitivity": 0.9}):
        assert client.get("/site-readiness/sample-size", params=params).status_code == 400, params


def test_sample_size_route_is_not_captured_as_a_job_id(client):
    assert client.get("/site-readiness/sample-size", params={"target_sensitivity": 0.9, "expected_sensitivity": 0.95}).status_code == 200
    assert client.get("/site-readiness/sr-doesnotexist").status_code == 404


# --- labels CSV ----------------------------------------------------------------------------------
def test_parse_labels():
    rows = sr.parse_labels("filename,label,score\na.png,1,0.9\nb.png,0,\n\nc.png,1,0.2\n")
    assert rows == [{"label": 1, "score": 0.9, "key": "a.png"}, {"label": 0, "score": None, "key": "b.png"}, {"label": 1, "score": 0.2, "key": "c.png"}]
    assert sr.parse_labels("") == []
    for bad in ("file,urgent\na,1\n", "label\nmaybe\n", "label,score\n1,high\n", "label,score\n1,1.7\n"):
        with pytest.raises(ValueError):
            sr.parse_labels(bad)
    assert sr._key("C:\\scans\\A.PNG") == "a" and sr._key("dir/sub/b.jpeg") == "b" and sr._key(None) is None


# --- verdict rules (model stubbed so every branch can be reached) -----------------------------------
class StubReal:
    """Stands in for the real model: returns chosen scores so the verdict logic can be tested in isolation."""
    has_drift_reference = False

    def __init__(self, ctx, scores, ood_rate=0.0, shift=0.0):
        self.ctx, self.scores, self.i = ctx, list(scores), 0
        self.ood_rate, self.shift = ood_rate, shift
        self.rng = np.random.default_rng(0)

    def analyse(self, pil):
        s = self.scores[self.i % len(self.scores)]
        self.i += 1
        val = self.ctx.predictions.val
        row = val.probs[int(self.rng.integers(len(val)))]          # a real reference row, so drift is small unless shifted
        probs = {f: float(min(1.0, p + self.shift)) for f, p in zip(self.ctx.predictions.findings, row)}
        return {"probs": probs, "composite_score": s, "ood_flag": bool(self.rng.random() < self.ood_rate), "embedding": np.zeros(4, np.float32)}


def run(ctx, monkeypatch, n_pos, caught, n_neg=60, **stub):
    cut = ctx.lanes().t_raise
    scores = [cut + 0.01] * caught + [max(0.0, cut - 0.2)] * (n_pos - caught) + [0.0] * n_neg
    monkeypatch.setattr(ctx.model, "active", "real")
    monkeypatch.setattr(ctx.model, "real", StubReal(ctx, scores, **stub))
    files = [(f"img{k}.png", cxr_png(64, 64, seed=k)) for k in range(n_pos + n_neg)]
    labels = "filename,label\n" + "".join(f"img{k}.png,{1 if k < n_pos else 0}\n" for k in range(n_pos + n_neg))
    return sr.evaluate(ctx, files, labels)


def test_ready_needs_a_passing_interval_and_a_clean_drift_check(tmp_path, monkeypatch):
    """B's drift rule only returns GO on a large sample that matches the reference, so use a large fixture."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    from .conftest import make_artifacts, settings_for
    with TestClient(create_app(settings_for(tmp_path, make_artifacts(tmp_path / "big", n=8000)))) as c:
        ctx = ctx_of(c)
        res = run(ctx, monkeypatch, n_pos=60, caught=60, n_neg=1140)
    assert res["sensitivity"]["ci_low"] >= 0.9 and res["drift"]["b_verdict"] == "GO", res["drift"]
    assert res["verdict"] == "ready"
    assert res["sensitivity"]["method"] == "clopper-pearson" and res["sensitivity"]["n_positive"] == 60 and res["target"] == 0.9
    assert abs(res["sensitivity"]["width"] - (res["sensitivity"]["ci_high"] - res["sensitivity"]["ci_low"])) < 1e-12


def test_small_sample_is_not_enough_evidence_even_when_every_scan_is_caught(client, monkeypatch):
    res = run(ctx_of(client), monkeypatch, n_pos=12, caught=12)
    assert res["sensitivity"]["point"] == 1.0 and res["verdict"] == "not_enough_evidence"
    assert "not evidence of poor performance" in res["note"]


def test_a_small_in_distribution_sample_is_never_called_not_ready_because_of_psi_noise(client, monkeypatch):
    """Images scored like the reference distribution: B's PSI cut-offs fire by chance. That must not become 'not_ready'."""
    ctx = ctx_of(client)
    for n_pos, n_neg in ((12, 60), (40, 20), (5, 30)):
        res = run(ctx, monkeypatch, n_pos=n_pos, caught=n_pos, n_neg=n_neg)
        assert res["verdict"] != "not_ready", res["drift"]
        if res["drift"]["b_verdict"] == "NO-GO":
            assert res["drift"]["b_verdict_counted_as"] == "WARNING" and res["drift"]["max_psi"] <= res["drift"]["chance_level_95"]
            assert "within what sample size alone produces" in res["note"]


def test_interval_straddling_the_target_is_not_enough_evidence_not_failure(client, monkeypatch):
    res = run(ctx_of(client), monkeypatch, n_pos=40, caught=37)             # 92.5% measured, lower bound below 90%
    s = res["sensitivity"]
    assert s["ci_low"] < 0.9 <= s["ci_high"] and res["verdict"] == "not_enough_evidence"


def test_not_ready_only_when_the_upper_bound_is_below_target(client, monkeypatch):
    res = run(ctx_of(client), monkeypatch, n_pos=60, caught=30)
    assert res["sensitivity"]["ci_high"] < 0.9 and res["verdict"] == "not_ready"
    assert "upper end of the sensitivity interval is below the target" in res["note"]


def test_drift_warning_downgrades_ready_to_use_with_caution(client, monkeypatch):
    res = run(ctx_of(client), monkeypatch, n_pos=60, caught=60, ood_rate=0.04)   # B: OOD rate >= 2.5% is a WARNING
    if res["drift"]["b_verdict"] == "WARNING":
        assert res["verdict"] == "use_with_caution" and res["drift"]["warning"] is True
    else:                                                                         # the random OOD draw happened to exceed 6%
        assert res["drift"]["b_verdict"] == "NO-GO" and res["verdict"] == "not_ready"


def test_real_drift_beyond_chance_is_not_ready(client, monkeypatch):
    ctx = ctx_of(client)
    res = run(ctx, monkeypatch, n_pos=60, caught=60, shift=0.4)                   # every probability shifted: PSI is huge
    d = res["drift"]
    assert d["b_verdict"] == "NO-GO" and d["b_verdict_counted_as"] == "NO-GO" and d["max_psi"] > d["chance_level_95"]
    assert res["verdict"] == "not_ready" and "image drift, not a measured drop" in res["note"]


def test_high_ood_rate_counts_only_with_enough_images(client, monkeypatch):
    ctx = ctx_of(client)
    many = run(ctx, monkeypatch, n_pos=40, caught=40, n_neg=40, ood_rate=0.5)
    assert many["drift"]["b_verdict"] == "NO-GO" and many["drift"]["b_verdict_counted_as"] == "NO-GO" and many["verdict"] == "not_ready"
    few = run(ctx, monkeypatch, n_pos=10, caught=10, n_neg=10, ood_rate=0.5)      # 20 images
    assert few["drift"]["b_verdict"] == "NO-GO" and few["verdict"] != "not_ready"


def test_never_ready_without_labels(client, monkeypatch):
    ctx = ctx_of(client)
    monkeypatch.setattr(ctx.model, "active", "real")
    monkeypatch.setattr(ctx.model, "real", StubReal(ctx, [0.9]))
    res = sr.evaluate(ctx, [(f"i{k}.png", cxr_png(64, 64, seed=k)) for k in range(60)], None)
    assert res["sensitivity"] is None and res["verdict"] == "not_enough_evidence"
    assert "No labels CSV was supplied" in res["note"]


def test_uploader_supplied_scores_are_labelled_and_cannot_reach_ready_without_a_drift_check(client):
    ctx = ctx_of(client)
    cut = ctx.lanes().t_raise
    labels = "label,score\n" + f"1,{min(0.999, cut + 0.01):.4f}\n" * 60 + "0,0.0\n" * 20
    res = sr.evaluate(ctx, [], labels)
    assert res["sensitivity"]["n_positive"] == 60 and res["sensitivity"]["ci_low"] >= 0.9
    assert res["verdict"] == "use_with_caution", "no images, so no drift check: never 'ready'"
    assert "supplied by the uploader" in res["note"] and "drift check could not run" in res["note"]


# --- endpoints -----------------------------------------------------------------------------------
def poll(c, job_id):
    for _ in range(200):
        r = c.get(f"/site-readiness/{job_id}").json()
        if r["status"] == "done":
            return r
    raise AssertionError("job did not finish")


def test_job_flow_without_labels(client, artifacts):
    files = [("files", (f"s{k}.png", cxr_png(seed=k), "image/png")) for k in range(3)] + [("files", ("notes.txt", b"hello", "text/plain"))]
    r = client.post("/site-readiness", files=files, data={"actor": "site lead"})
    assert r.status_code == 202 and set(r.json()) == {"job_id"}
    res = poll(client, r.json()["job_id"])
    assert {"status", "drift", "sensitivity", "target", "verdict"} <= set(res)
    assert {"score", "warning", "note"} <= set(res["drift"])
    assert res["sensitivity"] is None and res["verdict"] == "not_enough_evidence" and res["target"] == 0.9
    assert res["n_images"] == 3 and "1 file(s) could not be read" in res["note"]
    assert "not a proven predictor" in res["drift"]["note"]
    entry = [a for a in client.get("/audit").json() if a["action"] == "site_readiness_run"][0]
    assert entry["actor"] == "site lead" and r.json()["job_id"] in entry["detail"]


def test_job_in_replay_mode_does_not_invent_scores(client, artifacts):
    """The host runs no model here: images that are not saved demo images have no score, so no sensitivity."""
    files = [("files", (f"s{k}.png", cxr_png(seed=k), "image/png")) for k in range(4)]
    labels = "filename,label\n" + "".join(f"s{k}.png,1\n" for k in range(4))
    r = client.post("/site-readiness", files=files + [("labels", ("labels.csv", labels.encode(), "text/csv"))])
    res = poll(client, r.json()["job_id"])
    assert res["sensitivity"] is None and res["verdict"] == "not_enough_evidence"
    assert "does not run the model" in res["note"]
    # a byte-identical saved demo image does get its saved score
    files = [("files", ("DEMO-HIGH.png", demo_bytes(artifacts, "DEMO-HIGH"), "image/png"))]
    labels = "filename,label\nDEMO-HIGH.png,1\n"
    res = poll(client, client.post("/site-readiness", files=files + [("labels", ("l.csv", labels.encode(), "text/csv"))]).json()["job_id"])
    assert res["sensitivity"]["n_positive"] == 1 and res["sensitivity"]["point"] == 1.0 and res["verdict"] == "not_enough_evidence"


def test_job_input_validation(client):
    assert client.post("/site-readiness").status_code == 400
    img = ("files", ("a.png", cxr_png(), "image/png"))
    bad_csv = ("labels", ("l.csv", b"file,urgent\na,1\n", "text/csv"))
    r = client.post("/site-readiness", files=[img, bad_csv])
    assert r.status_code == 400 and "'label' column" in r.json()["detail"]
    assert client.post("/site-readiness", files=[img, ("labels", ("l.csv", b"\xff\xfe\x00bad", "text/csv"))]).status_code == 400
    too_many = [("files", (f"{k}.png", b"x", "image/png")) for k in range(101)]
    assert client.post("/site-readiness", files=too_many).status_code == 400
    assert [a for a in client.get("/audit").json() if a["action"] == "site_readiness_run"] == [], "rejected checks are not logged as runs"


def test_a_failing_job_reports_no_verdict_instead_of_crashing(client, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(sr, "evaluate", boom)
    r = client.post("/site-readiness", files=[("files", ("a.png", cxr_png(), "image/png"))])
    res = poll(client, r.json()["job_id"])
    assert res["verdict"] == "not_enough_evidence" and "could not be completed" in res["note"]
