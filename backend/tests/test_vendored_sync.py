"""The backend carries copies of teammates' code (the Docker build only sees backend/).
These tests fail if a copy drifts from its source, if the vocabularies stop agreeing, or if
Member B's own test-suite stops passing."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

BACKEND = Path(__file__).resolve().parents[1]
ROOT = BACKEND.parent
B_DIR = ROOT / "eval_safety" / "voxelsort_eval_safety"
A_DIR = ROOT / "ml" / "chesttriage_ml"
needs_sources = pytest.mark.skipif(not (B_DIR.is_dir() and A_DIR.is_dir()), reason="ml/ and eval_safety/ are not next to backend/ (Docker image)")


def _sync_module():
    spec = importlib.util.spec_from_file_location("sync_vendored", BACKEND / "scripts" / "sync_vendored.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@needs_sources
def test_vendored_copies_match_their_sources():
    assert _sync_module().stale() == [], "run: python scripts/sync_vendored.py"


@needs_sources
def test_only_imports_differ_in_the_evalkit_copy():
    sync = _sync_module()
    for name in sync.EVAL_MODULES:
        src = (B_DIR / f"{name}.py").read_text(encoding="utf-8").replace("\r\n", "\n").splitlines()
        dst = (BACKEND / "app" / "evalkit" / f"{name}.py").read_text(encoding="utf-8").splitlines()
        assert len(src) == len(dst)
        for a, b in zip(src, dst):
            if a != b:
                assert a.lstrip().startswith("from ") and b.replace("from .", "from ", 1) == a, (name, a, b)


@needs_sources
def test_static_copy_of_b_evidence_matches():
    copy = BACKEND / "app" / "static_evidence" / "locked_evidence.json"
    assert hashlib.sha256(copy.read_bytes()).hexdigest() == hashlib.sha256((B_DIR / "locked_evidence.json").read_bytes()).hexdigest()


def test_urgent_lists_of_a_and_b_agree():
    """A's policy decides the score; B's catalogue must not file any of A's urgent findings under ROUTINE, and vice versa."""
    from app.services import urgency
    from app.services.saved_predictions import DEFAULT_FINDINGS, DEFAULT_URGENT
    policy = json.loads((BACKEND / "model_artifacts" / "MANIFEST.json").read_text())["policy"]
    assert policy["findings"] == DEFAULT_FINDINGS and policy["urgent"] == DEFAULT_URGENT
    assert set(urgency.A_TO_B_RULE) == set(policy["findings"])
    for f in policy["findings"]:
        assert urgency.urgent_by_b(f) == (f in policy["urgent"]), f
    assert urgency.b_tier("consolidation") is None, "B has no separate rule for consolidation (documented)"
    assert urgency.TIER_TO_LANE == {"STAT": "raised", "URGENT": "raised", "ROUTINE": "normal"}


def test_evalkit_is_importable_as_a_package():
    from app.evalkit.calibration import select_threshold_at_sensitivity
    from app.evalkit.protocol import calculate_required_sample_size, compute_binary_metrics
    from app.evalkit.queue_simulator import QueueSimulator
    from app.evalkit.site_readiness import SiteReadinessAssessor
    from app.evalkit.urgency_mapping import PriorityLevel, UrgencyClassifier
    assert callable(select_threshold_at_sensitivity) and callable(compute_binary_metrics) and callable(calculate_required_sample_size)
    assert QueueSimulator and SiteReadinessAssessor and UrgencyClassifier and PriorityLevel.STAT.value == "STAT"


@needs_sources
def test_member_b_own_tests_still_pass():
    """B's files are untouched; their tests use flat imports, so they run from B's folder."""
    r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q", "-p", "no:cacheprovider"], cwd=B_DIR, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert " passed" in r.stdout and "failed" not in r.stdout


@needs_sources
def test_member_b_tests_also_pass_against_the_vendored_copy(tmp_path):
    """Run B's test files against app.evalkit by aliasing the flat module names to the package copies."""
    shim = tmp_path / "conftest.py"
    shim.write_text(
        "import importlib, sys\n"
        f"sys.path.insert(0, {str(BACKEND)!r})\n"
        "for name in ('urgency_mapping', 'protocol', 'calibration', 'queue_simulator', 'site_readiness', 'shortcut_audit'):\n"
        "    sys.modules[name] = importlib.import_module('app.evalkit.' + name)\n", encoding="utf-8")
    for t in (B_DIR / "tests").glob("test_*.py"):
        (tmp_path / t.name).write_bytes(t.read_bytes())
    r = subprocess.run([sys.executable, "-m", "pytest", str(tmp_path), "-q", "-p", "no:cacheprovider"], cwd=tmp_path, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
