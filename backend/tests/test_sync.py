"""The backend ships its own copy of B's locked_evidence.json (the Docker build only sees backend/).
These tests fail if the copy drifts from B's file, or if the urgency split drifts from B's rules."""
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
B_DIR = ROOT / "eval_safety" / "voxelsort_eval_safety"
COPY = Path(__file__).resolve().parents[1] / "app" / "static_evidence" / "locked_evidence.json"

pytestmark = pytest.mark.skipif(not B_DIR.is_dir(), reason="eval_safety/ is not next to backend/ (Docker build)")


def test_evidence_copy_matches_b():
    assert hashlib.sha256(COPY.read_bytes()).hexdigest() == hashlib.sha256((B_DIR / "locked_evidence.json").read_bytes()).hexdigest()


def test_routine_findings_match_b_urgency_mapping():
    sys.path.insert(0, str(B_DIR))
    try:
        spec = importlib.util.spec_from_file_location("b_urgency_mapping", B_DIR / "urgency_mapping.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules["b_urgency_mapping"] = mod
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(str(B_DIR))
    from app.ranking import ROUTINE_FINDINGS, norm_name
    b_routine = {norm_name(n) for n, r in mod.CLINICAL_FINDING_RULES.items() if r.target_priority.value == "ROUTINE"}
    assert b_routine == ROUTINE_FINDINGS
