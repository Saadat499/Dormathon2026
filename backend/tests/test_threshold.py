"""Safety vs Speed: preview changes nothing, a change needs confirmation and respects the sensitivity floor."""
from __future__ import annotations

import numpy as np
from fastapi.testclient import TestClient

from app.main import create_app
from app.models import SavedPrediction, Setting

from .conftest import ctx_of, settings_for

PREVIEW_KEYS = {"abstention_rate", "sensitivity", "false_alarm_rate", "n", "n_positive"}


def test_default_threshold_is_the_highest_value_meeting_the_target(client):
    t = client.get("/threshold").json()
    assert {"value", "min_sensitivity_target"} <= set(t) and t["min_sensitivity_target"] == 0.9
    ctx = ctx_of(client)
    v = t["value"]
    assert ctx.predictions.sensitivity_at(v) >= 0.9
    assert ctx.predictions.sensitivity_at(round(v + 0.01, 2)) < 0.9, "one step higher would break the floor"
    assert t["max_allowed_value"] == v and t["data_is_real"] is False and t["data_source"] == "synthetic_test_fixture"


def test_preview_matches_an_independent_calculation_on_the_validation_split(client):
    ctx = ctx_of(client)
    val, test = ctx.predictions.val, ctx.predictions.test
    for value in (0.2, 0.5, 0.8):
        p = client.post("/threshold/preview", json={"value": value}).json()
        assert PREVIEW_KEYS <= set(p)
        cut = float(np.quantile(val.scores, value))
        low = float(np.quantile(val.scores, value - 0.1))
        raised = (val.scores >= cut) & ~val.ood
        assert abs(p["sensitivity"] - raised[val.labels == 1].mean()) < 1e-12
        assert abs(p["false_alarm_rate"] - raised[val.labels == 0].mean()) < 1e-12
        unsure = val.ood | ((val.scores >= low) & (val.scores < cut))
        assert abs(p["abstention_rate"] - unsure.mean()) < 1e-12
        assert p["n"] == len(val) and p["n_positive"] == int((val.labels == 1).sum())
        assert abs(p["score_cutoff"] - cut) < 1e-12
        # the held-out split is reported separately and is NOT what the top-level numbers use
        t_raised = (test.scores >= cut) & ~test.ood
        assert abs(p["held_out_test"]["sensitivity"] - t_raised[test.labels == 1].mean()) < 1e-12
        assert p["held_out_test"]["n"] == len(test)
        assert p["data_is_real"] is False and "NOT REAL MODEL OUTPUT" in p["note"]


def test_preview_changes_nothing(client):
    before = client.get("/threshold").json()
    n_audit = len(client.get("/audit").json())
    for v in (0.05, 0.5, 0.95):
        assert client.post("/threshold/preview", json={"value": v}).status_code == 200
    assert client.get("/threshold").json() == before
    assert len(client.get("/audit").json()) == n_audit


def test_sensitivity_falls_and_false_alarms_fall_as_the_threshold_rises(client):
    rows = [client.post("/threshold/preview", json={"value": v / 20}).json() for v in range(1, 20)]
    sens = [r["sensitivity"] for r in rows]
    far = [r["false_alarm_rate"] for r in rows]
    assert sens == sorted(sens, reverse=True) and far == sorted(far, reverse=True)
    assert all(0.0 <= r["abstention_rate"] <= 1.0 for r in rows)


def test_change_needs_confirm(client):
    current = client.get("/threshold").json()["value"]
    lower = round(current - 0.05, 2)
    for body in ({"value": lower}, {"value": lower, "confirm": False}, {"value": lower, "confirm": "yes please"}):
        r = client.put("/threshold", json=body)
        assert r.status_code == 400 and "detail" in r.json()
    assert client.get("/threshold").json()["value"] == current
    assert [a for a in client.get("/audit").json() if a["action"] == "threshold_change"] == []


def test_change_is_refused_below_the_sensitivity_floor(client):
    current = client.get("/threshold").json()["value"]
    r = client.put("/threshold", json={"value": round(current + 0.05, 2), "confirm": True, "actor": "lead"})
    assert r.status_code == 400 and "below the minimum target of 90%" in r.json()["detail"]
    assert client.get("/threshold").json()["value"] == current
    assert [a for a in client.get("/audit").json() if a["action"] == "threshold_change"] == []


def test_confirmed_change_is_applied_logged_and_reranks_the_queue(seeded_client):
    c = seeded_client
    ctx = ctx_of(c)
    with ctx.db.session() as s:                                   # a scan whose score sits just under the current cut-off
        from app.models import Study
        lanes = ctx.lanes()
        row = s.get(Study, "DEMO-LOW")
        row.composite_score = lanes.t_raise - 1e-6
    assert c.get("/studies/DEMO-LOW").json()["priority"] == "unsure"
    current = c.get("/threshold").json()["value"]
    new = round(current - 0.1, 2)
    r = c.put("/threshold", json={"value": new, "confirm": True, "actor": "clinical lead"})
    assert r.status_code == 200 and r.json()["value"] == new and r.json()["min_sensitivity_target"] == 0.9
    assert c.get("/threshold").json()["value"] == new
    assert c.get("/studies/DEMO-LOW").json()["priority"] == "raised", "priority is derived at read time from the new threshold"
    entry = [a for a in c.get("/audit").json() if a["action"] == "threshold_change"][0]
    assert entry["actor"] == "clinical lead" and f"to {new:.2f}" in entry["detail"] and entry["study_id"] is None
    assert entry["data"]["old"] == current and entry["data"]["new"] == new and entry["data"]["confirmed"] is True


def test_bad_values_are_rejected(client):
    for v in (0, 1, -0.2, 1.5, "high", None, float("nan")):
        body = {"value": v} if v == v else '{"value": NaN}'
        r = client.post("/threshold/preview", json=body) if isinstance(body, dict) else \
            client.post("/threshold/preview", content=body, headers={"Content-Type": "application/json"})
        assert r.status_code == 400, v
    assert client.put("/threshold", json={"value": 7, "confirm": True}).status_code == 400
    assert client.post("/threshold/preview", json={}).status_code == 400


def test_without_saved_predictions_threshold_endpoints_say_so(bare_client):
    c = bare_client
    assert c.get("/threshold").json()["value"] is None
    for r in (c.post("/threshold/preview", json={"value": 0.5}), c.put("/threshold", json={"value": 0.5, "confirm": True})):
        assert r.status_code == 503 and "not available" in r.json()["detail"]
    h = c.get("/health").json()
    assert h["degraded"] is True and h["saved_predictions"]["available"] is False and h["saved_predictions"]["source"] == "none"


def test_a_stored_threshold_that_breaks_the_floor_is_reset_at_startup(tmp_path, artifacts):
    st = settings_for(tmp_path, artifacts)
    with TestClient(create_app(st)) as c:
        good = c.get("/threshold").json()["value"]
        ctx = ctx_of(c)
        with ctx.db.session() as s:
            s.get(Setting, "threshold").value = 0.95                 # e.g. written before the target was raised
    with TestClient(create_app(st)) as c:
        assert c.get("/threshold").json()["value"] == good
        entry = [a for a in c.get("/audit").json() if a["action"] == "threshold_change"][0]
        assert entry["actor"] == "system" and "no longer meets the minimum sensitivity target" in entry["detail"]


def test_target_comes_from_the_environment_setting(tmp_path, artifacts):
    with TestClient(create_app(settings_for(tmp_path, artifacts, min_sensitivity_target=0.5))) as c:
        loose = c.get("/threshold").json()
    assert loose["min_sensitivity_target"] == 0.5
    with TestClient(create_app(settings_for(tmp_path / "b", artifacts))) as c:
        strict = c.get("/threshold").json()
    assert loose["value"] > strict["value"]


def test_saved_predictions_table_mirrors_the_file(client):
    ctx = ctx_of(client)
    with ctx.db.session() as s:
        from sqlalchemy import func, select
        n = s.scalar(select(func.count()).select_from(SavedPrediction))
        splits = dict(s.execute(select(SavedPrediction.split, func.count()).group_by(SavedPrediction.split)).all())
        one = s.scalars(select(SavedPrediction).limit(1)).one()
    assert n == len(ctx.predictions.val) + len(ctx.predictions.test) == 800
    assert splits == {"val": 400, "test": 400}
    assert one.source == "synthetic_test_fixture" and set(one.probs) == set(ctx.predictions.findings)
    assert one.true_label_any_urgent in (0, 1)
