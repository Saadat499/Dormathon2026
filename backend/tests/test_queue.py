"""The ranking rule on its own: properties that must hold for ANY queue, checked on many random queues."""
from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone

import pytest

from app.services import queue
from app.services.queue import Lanes, QueueItem

T0 = datetime(2026, 10, 11, 12, 0, tzinfo=timezone.utc)
LANES = Lanes(t_raise=0.5, t_unsure=0.35)


def random_queue(rng: random.Random, n: int, max_wait: int = 120) -> list[QueueItem]:
    items = []
    for k in range(n):
        waited = rng.choice([0, 1, 5, 30, 60, max_wait - 1, max_wait, max_wait + 1, 300]) + rng.random() * rng.choice([0, 0, 3])
        items.append(QueueItem(
            study_id=f"S{k}", seq=k, uploaded_at=T0 - timedelta(minutes=waited), clinician_urgent=rng.random() < 0.15,
            priority=rng.choice(queue.PRIORITIES), waited_minutes=waited, reviewed=rng.random() < 0.1))
    rng.shuffle(items)
    return items


@pytest.mark.parametrize("seed", range(60))
@pytest.mark.parametrize("mode", ["model", "fcfs"])
def test_nothing_is_dropped_or_hidden(seed, mode):
    rng = random.Random(seed)
    items = random_queue(rng, rng.randint(0, 40))
    out = queue.rank(items, mode, 120)
    assert sorted(i.study_id for i, _ in out) == sorted(i.study_id for i in items)        # a permutation
    open_ranks = [r for i, r in out if not i.reviewed]
    assert open_ranks == list(range(1, len(open_ranks) + 1))                                # ranks 1..n, no gaps
    assert all(r is None for i, r in out if i.reviewed)
    flags = [i.reviewed for i, _ in out]
    assert flags == sorted(flags), "reviewed scans come after every open scan"


@pytest.mark.parametrize("seed", range(60))
def test_model_mode_group_order(seed):
    """clinician request > wait cap > raised > unsure > normal > oldest first."""
    rng = random.Random(1000 + seed)
    out = [i for i, _ in queue.rank(random_queue(rng, 40), "model", 120) if not i.reviewed]

    def group(i):
        return (0 if i.clinician_urgent else 1, 0 if i.waited_minutes >= 120 else 1, queue.PRIORITY_ORDER[i.priority])
    assert [group(i) for i in out] == sorted(group(i) for i in out)
    for a, b in zip(out, out[1:]):
        if group(a) == group(b):
            assert (a.uploaded_at, a.seq) <= (b.uploaded_at, b.seq)                         # oldest first inside a group


@pytest.mark.parametrize("seed", range(40))
def test_unsure_never_ranks_below_normal_in_the_same_group(seed):
    rng = random.Random(2000 + seed)
    out = [i for i, _ in queue.rank(random_queue(rng, 40), "model", 120) if not i.reviewed]
    for pos, i in enumerate(out):
        if i.priority != "normal":
            continue
        for later in out[pos + 1:]:
            same_group = later.clinician_urgent == i.clinician_urgent and (later.waited_minutes >= 120) == (i.waited_minutes >= 120)
            assert not (same_group and later.priority in ("unsure", "raised")), "a normal scan is ahead of an unsure/raised one"


@pytest.mark.parametrize("seed", range(40))
def test_clinician_requests_are_always_the_first_group(seed):
    rng = random.Random(3000 + seed)
    for mode in ("model", "fcfs"):
        out = [i for i, _ in queue.rank(random_queue(rng, 30), mode, 120) if not i.reviewed]
        flags = [i.clinician_urgent for i in out]
        assert flags == sorted(flags, reverse=True)


def test_wait_cap_promotes_a_normal_scan_above_raised():
    fresh_raised = QueueItem("R", 2, T0 - timedelta(minutes=1), False, "raised", 1)
    old_normal = QueueItem("N", 1, T0 - timedelta(minutes=121), False, "normal", 121)
    assert [i.study_id for i, _ in queue.rank([fresh_raised, old_normal], "model", 120)] == ["N", "R"]
    just_under = QueueItem("N", 1, T0 - timedelta(minutes=119), False, "normal", 119)
    assert [i.study_id for i, _ in queue.rank([fresh_raised, just_under], "model", 120)] == ["R", "N"]
    # a clinician request still beats a capped scan
    clin = QueueItem("C", 3, T0, True, "normal", 0)
    assert [i.study_id for i, _ in queue.rank([fresh_raised, old_normal, clin], "model", 120)] == ["C", "N", "R"]


@pytest.mark.parametrize("seed", range(40))
def test_fcfs_ignores_the_model_the_cap_and_overrides(seed):
    rng = random.Random(4000 + seed)
    items = random_queue(rng, 30)
    out = [i for i, _ in queue.rank(items, "fcfs", 120) if not i.reviewed]
    expect = sorted((i for i in items if not i.reviewed), key=lambda i: (not i.clinician_urgent, i.uploaded_at, i.seq))
    assert [i.study_id for i in out] == [i.study_id for i in expect]
    # changing every priority changes nothing in fcfs mode
    flipped = [QueueItem(i.study_id, i.seq, i.uploaded_at, i.clinician_urgent, "normal", i.waited_minutes, i.reviewed) for i in items]
    assert [i.study_id for i, _ in queue.rank(flipped, "fcfs", 120)] == [i.study_id for i, _ in queue.rank(items, "fcfs", 120)]


def test_ties_are_stable_and_repeatable():
    same_time = [QueueItem(f"S{k}", k, T0, False, "raised", 10) for k in range(12)]
    rng = random.Random(7)
    for _ in range(20):
        shuffled = same_time[:]
        rng.shuffle(shuffled)
        assert [i.study_id for i, _ in queue.rank(shuffled, "model", 120)] == [f"S{k}" for k in range(12)]


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        queue.rank([], "random", 120)


# --- lane derivation ---------------------------------------------------------------------------
@pytest.mark.parametrize("score", [None, float("nan"), 0.0, 0.2, 0.34, 0.99])
def test_ood_forces_unsure_whatever_the_score(score):
    assert queue.model_priority(score, True, LANES) == "unsure"
    assert queue.confidence_band(score, True, LANES) == "none"


def test_missing_score_or_cutoffs_is_unsure_never_normal():
    assert queue.model_priority(None, False, LANES) == "unsure"
    assert queue.model_priority(float("nan"), False, LANES) == "unsure"
    assert queue.model_priority(0.01, False, Lanes(None, None)) == "unsure"
    assert queue.effective_priority(None, None) == "unsure"


def test_lanes_from_score():
    assert queue.model_priority(0.5, False, LANES) == "raised"
    assert queue.model_priority(0.49, False, LANES) == "unsure"
    assert queue.model_priority(0.35, False, LANES) == "unsure"
    assert queue.model_priority(0.349, False, LANES) == "normal"
    assert queue.confidence_band(0.4, False, LANES) == "low"
    assert queue.confidence_band(0.9, False, LANES) == "high"
    assert queue.confidence_band(0.1, False, LANES) == "high"


def test_the_model_can_only_move_a_scan_up_relative_to_arrival_order():
    """With no human input, a scan is never read later than every scan that arrived after it AND has a lower lane."""
    rng = random.Random(11)
    for _ in range(50):
        items = [QueueItem(f"S{k}", k, T0 - timedelta(minutes=100 - k), False, rng.choice(queue.PRIORITIES), 100 - k) for k in range(25)]
        order = {i.study_id: n for i, n in queue.rank(items, "model", 10_000)}
        for a in items:
            for b in items:
                if a.seq < b.seq and queue.PRIORITY_ORDER[a.priority] <= queue.PRIORITY_ORDER[b.priority]:
                    assert order[a.study_id] < order[b.study_id]


# --- reasons and overrides ---------------------------------------------------------------------
def test_reason_precedence():
    assert queue.priority_reason(True, True, "raised") == "clinician_request"
    assert queue.priority_reason(False, True, "raised") == "wait_cap"
    assert queue.priority_reason(False, False, "normal") == "coordinator_override"
    assert queue.priority_reason(False, False, None) == "model"


def test_override_replaces_the_model_lane_only():
    assert queue.effective_priority("normal", "raised") == "raised"
    assert queue.effective_priority("raised", "normal") == "normal"        # only a human can lower a scan
    assert queue.effective_priority("raised", None) == "raised"
    assert queue.effective_priority("raised", "bogus") == "raised"
    # override vs the higher rules: clinician request and the wait cap still come first
    overridden_low = QueueItem("O", 1, T0 - timedelta(minutes=200), False, "normal", 200)   # human set it to normal, but it is past the cap
    raised = QueueItem("R", 2, T0, False, "raised", 0)
    assert [i.study_id for i, _ in queue.rank([raised, overridden_low], "model", 120)] == ["O", "R"]
    assert not queue.is_over_cap(500, 120, "fcfs")
