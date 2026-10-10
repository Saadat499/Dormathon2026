from app.ranking import derive_priority, rank_studies, urgency_score


def S(i, prio, up, urgent=False, cap=False, status="awaiting_review"):
    return {"study_id": i, "effective_priority": prio, "uploaded_at": up, "_seq": 0, "clinician_urgent": urgent,
            "priority_reason": "wait_cap" if cap else "model", "status": status, "rank": None}


def test_derive_priority_never_normal_when_unknown():
    assert derive_priority(None, False, 0.5) == "unsure"
    assert derive_priority(float("nan"), False, 0.5) == "unsure"
    assert derive_priority(0.01, True, 0.5) == "unsure"      # OOD never normal
    assert derive_priority(0.6, True, 0.5) == "unsure"       # OOD never auto-raised or cleared
    assert derive_priority(0.5, False, 0.5) == "raised"
    assert derive_priority(0.40, False, 0.5) == "unsure"
    assert derive_priority(0.30, False, 0.5) == "normal"


def test_urgency_score_unknown_finding_counts_as_urgent_and_routine_does_not():
    assert urgency_score([{"name": "Atelectasis", "probability": 0.99}]) == 0.0
    assert urgency_score([{"name": "mystery finding", "probability": 0.7}]) == 0.7
    assert urgency_score([]) is None


def test_queue_rule_order():
    items = [
        S("normal-old", "normal", "2026-01-01T00:00:01Z"),
        S("unsure", "unsure", "2026-01-01T00:00:02Z"),
        S("raised-new", "raised", "2026-01-01T00:00:09Z"),
        S("raised-old", "raised", "2026-01-01T00:00:03Z"),
        S("capped-normal", "normal", "2026-01-01T00:00:05Z", cap=True),
        S("clinician-normal", "normal", "2026-01-01T00:00:08Z", urgent=True),
        S("done", "raised", "2026-01-01T00:00:00Z", status="reviewed"),
    ]
    ranked = rank_studies(items, "model")
    assert [s["study_id"] for s in ranked] == ["clinician-normal", "capped-normal", "raised-old", "raised-new", "unsure", "normal-old", "done"]
    assert [s["rank"] for s in ranked] == [1, 2, 3, 4, 5, 6, None]


def test_fcfs_ignores_model_but_not_a_clinician_request():
    items = [S("a", "normal", "2026-01-01T00:00:01Z"), S("b", "raised", "2026-01-01T00:00:02Z"), S("c", "normal", "2026-01-01T00:00:03Z", urgent=True)]
    assert [s["study_id"] for s in rank_studies(items, "fcfs")] == ["c", "a", "b"]


def test_unsure_always_above_normal():
    items = [S("n", "normal", "2026-01-01T00:00:01Z"), S("u", "unsure", "2026-01-01T00:00:59Z")]
    assert [s["study_id"] for s in rank_studies(items, "model")] == ["u", "n"]
