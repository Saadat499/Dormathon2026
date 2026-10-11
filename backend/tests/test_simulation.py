"""Queue Race: contract shape, determinism, input limits, and that the numbers come from saved predictions."""
from __future__ import annotations

from app.services import queue, simulation

from .conftest import ctx_of

BODY = {"arrivals_per_hour": 20, "reads_per_hour": 18, "urgent_share": 0.15, "hours": 8, "seed": 1}
ARM = {"urgent_wait_min", "other_extra_wait_min", "missed_urgent", "wrong_flags"}


def test_shape_follows_the_contract(client):
    r = client.post("/simulations/queue-race", json=BODY)
    assert r.status_code == 200
    j = r.json()
    assert {"assumptions", "fcfs", "voxelsort", "read_order"} <= set(j)
    assert ARM <= set(j["fcfs"]) and ARM <= set(j["voxelsort"])
    assert j["fcfs"]["missed_urgent"] is None and j["fcfs"]["wrong_flags"] is None, "FCFS has no model"
    assert isinstance(j["voxelsort"]["missed_urgent"], int) and isinstance(j["voxelsort"]["wrong_flags"], int)
    assert set(j["read_order"]) == {"fcfs", "voxelsort"}
    for row in j["read_order"]["fcfs"] + j["read_order"]["voxelsort"]:
        assert {"id", "wait_min", "truly_urgent"} <= set(row) and row["id"].startswith("SIM-") and isinstance(row["truly_urgent"], bool)
    a = j["assumptions"]
    for k, v in BODY.items():
        assert a[k] == v, "the assumptions are echoed back"
    assert a["readers"] == 1 and "not clinical proof" in a["note"] and a["max_wait_minutes"] == 120
    assert j["data_is_real"] is False and "NOT REAL MODEL OUTPUT" in a["note"], "the fixture must be labelled"
    assert j["n_cases"] > 0 and 0 <= j["n_urgent"] <= j["n_cases"]


def test_same_seed_same_result_different_seed_different_result(client):
    a = client.post("/simulations/queue-race", json=BODY).json()
    b = client.post("/simulations/queue-race", json=BODY).json()
    c = client.post("/simulations/queue-race", json={**BODY, "seed": 2}).json()
    assert a == b
    assert (a["n_cases"], a["read_order"]) != (c["n_cases"], c["read_order"])


def test_cases_are_drawn_from_the_held_out_saved_predictions(client):
    ctx = ctx_of(client)
    stream = simulation.build_stream(ctx, 30, 20, 0.3, 6, seed=5)
    test = ctx.predictions.test
    label = dict(zip(test.study_ids, test.labels))
    score = dict(zip(test.study_ids, test.scores))
    ood = dict(zip(test.study_ids, test.ood))
    lanes = ctx.lanes()
    assert len(stream) > 50
    for c in stream:
        sid = c["source_study"]
        assert sid in label, "every simulated case is a real row of the test split"
        assert bool(label[sid]) == c["truly_urgent"]
        assert c["lane"] == queue.model_priority(float(score[sid]), bool(ood[sid]), lanes)
    arrivals = [c["arrival_time"] for c in stream]
    assert arrivals == sorted(arrivals) and arrivals[-1] < 6 * 60


def test_miss_and_wrong_flag_counts_match_the_stream(client):
    ctx = ctx_of(client)
    j = client.post("/simulations/queue-race", json={**BODY, "seed": 9}).json()
    stream = simulation.build_stream(ctx, 20, 18, 0.15, 8, seed=9)
    assert j["n_cases"] == len(stream) and j["n_urgent"] == sum(c["truly_urgent"] for c in stream)
    assert j["voxelsort"]["missed_urgent"] == sum(c["truly_urgent"] and c["lane"] == "normal" for c in stream)
    assert j["voxelsort"]["wrong_flags"] == sum((not c["truly_urgent"]) and c["lane"] == "raised" for c in stream)


def test_fcfs_arm_agrees_with_an_independent_first_come_first_served_run(client):
    """Member B's FIFO engine and this module's event loop must describe the same queue."""
    ctx = ctx_of(client)
    j = client.post("/simulations/queue-race", json={**BODY, "seed": 4}).json()
    stream = simulation.build_stream(ctx, 20, 18, 0.15, 8, seed=4)
    clock, waits = 0.0, []
    for c in stream:
        start = max(clock, c["arrival_time"])
        waits.append((c["truly_urgent"], start - c["arrival_time"]))
        clock = start + c["read_time_needed"]
    urgent = [w for u, w in waits if u]
    assert abs(j["fcfs"]["urgent_wait_min"] - sum(urgent) / len(urgent)) < 0.06
    fifo_like = simulation.run_live_rule(stream, 10_000, mode="fcfs")
    assert [c["study_id"] for c in fifo_like] == [c["study_id"] for c in stream]
    assert max(abs(a["wait_time_minutes"] - w) for a, (_, w) in zip(fifo_like, waits)) < 1e-9


def test_live_rule_reads_raised_before_normal_and_honours_the_wait_cap():
    mk = lambda i, t, lane, read=10.0: {"study_id": f"S{i}", "idx": i, "arrival_time": t, "read_time_needed": read, "lane": lane, "truly_urgent": lane == "raised"}  # noqa: E731
    stream = [mk(0, 0.0, "normal"), mk(1, 1.0, "normal"), mk(2, 2.0, "raised"), mk(3, 3.0, "unsure")]
    order = [c["study_id"] for c in simulation.run_live_rule(stream, max_wait_minutes=10_000)]
    assert order == ["S0", "S2", "S3", "S1"], "S0 is read first (nothing else has arrived), then raised, unsure, normal"
    # with a short cap the waiting normal scan is promoted ahead of the raised one
    stream = [mk(0, 0.0, "normal", 30.0), mk(1, 1.0, "normal"), mk(2, 29.0, "raised")]
    assert [c["study_id"] for c in simulation.run_live_rule(stream, max_wait_minutes=20)] == ["S0", "S1", "S2"]
    assert [c["study_id"] for c in simulation.run_live_rule(stream, max_wait_minutes=60)] == ["S0", "S2", "S1"]


def test_inputs_are_validated_and_capped(client):
    bad = [
        {**BODY, "hours": 25}, {**BODY, "hours": 0}, {**BODY, "arrivals_per_hour": 0}, {**BODY, "arrivals_per_hour": 201},
        {**BODY, "reads_per_hour": -1}, {**BODY, "urgent_share": 0}, {**BODY, "urgent_share": 1}, {**BODY, "urgent_share": "lots"},
        {**BODY, "seed": -1}, {**BODY, "seed": 1.5}, {**BODY, "hours": True}, {k: v for k, v in BODY.items() if k != "hours"},
    ]
    for body in bad:
        r = client.post("/simulations/queue-race", json=body)
        assert r.status_code == 400 and isinstance(r.json()["detail"], str), body
    assert client.post("/simulations/queue-race", json=[1, 2]).status_code == 400
    assert client.post("/simulations/queue-race", content="not json", headers={"Content-Type": "application/json"}).status_code == 400
    too_many = client.post("/simulations/queue-race", json={**BODY, "arrivals_per_hour": 200, "hours": 24})
    assert too_many.status_code == 400 and "Too many scans" in too_many.json()["detail"]
    import time
    t = time.perf_counter()                                            # worst case: the queue only ever grows
    r = client.post("/simulations/queue-race", json={**BODY, "arrivals_per_hour": 100, "reads_per_hour": 1, "hours": 15})
    assert r.status_code == 200 and r.json()["n_cases"] <= 1500
    assert time.perf_counter() - t < 30, "the largest allowed request must finish quickly enough not to hang the server"


def test_without_saved_predictions_the_race_refuses_to_invent_numbers(bare_client):
    r = bare_client.post("/simulations/queue-race", json=BODY)
    assert r.status_code == 503 and "cannot run" in r.json()["detail"]


def test_simulation_writes_no_audit_entry_and_changes_no_setting(client):
    before = (client.get("/audit").json(), client.get("/threshold").json(), client.get("/queue/settings").json())
    client.post("/simulations/queue-race", json=BODY)
    assert (client.get("/audit").json(), client.get("/threshold").json(), client.get("/queue/settings").json()) == before
