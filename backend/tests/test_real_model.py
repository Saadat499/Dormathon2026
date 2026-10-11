"""Member A's REAL model, end to end. Skipped when the model files are not installed
(`python scripts/fetch_model.py` installs them). Nothing here is synthetic: the images are public NIH
ChestX-ray14 demo images from A's bundle and the expected numbers are A's own saved outputs."""
from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from .conftest import REAL_ARTIFACTS, ctx_of, wait_done

DEMO = REAL_ARTIFACTS / "demo"


def demo_studies() -> list[dict]:
    return json.loads((DEMO / "demo_studies.json").read_text(encoding="utf-8"))["studies"]


def image_bytes(rec: dict) -> bytes:
    return (REAL_ARTIFACTS / rec["image"]).read_bytes()


def test_selected_model_is_the_frozen_v2_and_files_are_verified(real_client):
    ctx = ctx_of(real_client)
    manifest = json.loads((REAL_ARTIFACTS / "MANIFEST.json").read_text())
    source = json.loads((REAL_ARTIFACTS / "SOURCE.json").read_text())
    assert ctx.model.active == "real" and ctx.model.verify_runtime_files() == []
    assert ctx.model.version == manifest["model_version"] == source["model_version"] == "partA-0.1.0+4ca280b3cd"
    assert source["artifacts_folder"] == "artifacts_v2_imagenet_fullnih" and source["data"] == "real"
    assert manifest["backbone"] == "densenet121-onnx" and ctx.model.real.provider == "CPUExecutionProvider"
    assert ctx.model.findings == manifest["policy"]["findings"] and ctx.model.urgent == manifest["policy"]["urgent"]
    h = real_client.get("/health").json()
    assert h["model_mode"] == "real" and h["model"]["runs_a_trained_model"] is True and h["degraded"] is False
    assert h["saved_predictions"]["real"] is True and h["saved_predictions"]["n_reference_val"] == 11341 and h["saved_predictions"]["n_held_out_test"] == 21845


def test_live_inference_reproduces_the_saved_outputs(real_client):
    """The whole chain (preprocessing, backbone, head, OOD detector) must match what A exported."""
    model = ctx_of(real_client).model
    checked = 0
    for rec in demo_studies():
        out = model.predict(image_bytes(rec))
        assert out.source == "real" and out.ood_flag == rec["ood_flag"], rec["study_id"]
        if rec["ood_flag"]:
            assert out.findings == [] and out.composite_score is None and out.cam is None
            continue
        got = {f["name"]: f["probability"] for f in out.findings}
        assert set(got) == set(rec["findings"])
        assert max(abs(got[k] - v) for k, v in rec["findings"].items()) < 2e-4, rec["study_id"]
        assert abs(out.composite_score - rec["composite_score"]) < 2e-4
        assert out.cam_finding == rec["top_urgent_finding"] and out.cam.shape == (224, 224) and 0.0 <= float(out.cam.min()) <= float(out.cam.max()) <= 1.0
        checked += 1
    assert checked >= 40


def test_backend_forward_pass_equals_member_a_engine(real_client):
    """`RealModel.predict` composes A's building blocks; it must agree with A's own `engine.predict`."""
    real = ctx_of(real_client).model.real
    for rec in demo_studies()[:10] + [r for r in demo_studies() if r["ood_flag"]][:4]:
        pil = Image.open(REAL_ARTIFACTS / rec["image"])
        pil.load()
        a = real.engine.predict(pil, study_id=rec["study_id"])
        b = real.predict(pil)
        assert a["ood_flag"] == b.ood_flag and a["model_version"] == b.model_version
        assert a["findings"] == {f["name"]: f["probability"] for f in b.findings}
        assert a["composite_score"] == b.composite_score
        if not b.ood_flag:
            cam, finding = real.engine.cam(pil)
            assert finding == b.cam_finding and np.allclose(cam, b.cam)


def test_upload_flow_with_real_inference(real_client):
    c = real_client
    rec = next(r for r in demo_studies() if r["category"] == "nih_confident_hit")
    r = c.post("/studies", files={"file": ("hit.png", image_bytes(rec), "image/png")}, data={"actor": "real-test"})
    assert r.status_code == 202 and r.json()["status"] == "processing"
    s = wait_done(c, r.json()["study_id"])
    assert s["model_source"] == "real" and s["model_version"] == "partA-0.1.0+4ca280b3cd"
    assert {f["name"]: f["probability"] for f in s["findings"]} == rec["findings"]
    assert s["priority"] == "raised" and s["status"] == "awaiting_review" and s["confidence_band"] == "high"
    assert s["heatmap_path"] == f"/studies/{s['study_id']}/heatmap"
    heat = c.get(s["heatmap_path"])
    assert heat.status_code == 200 and heat.headers["content-type"] == "image/png"
    import io
    overlay, shown = Image.open(io.BytesIO(heat.content)), Image.open(io.BytesIO(c.get(f"/studies/{s['study_id']}/image").content))
    assert overlay.mode == "RGBA" and overlay.size == shown.size, "the overlay lines up with the displayed image"
    assert not s["plain_language"].startswith(("Replayed", "Demo record", "FAKE"))
    # override -> audit -> review
    o = c.post(f"/studies/{s['study_id']}/override", json={"priority": "unsure", "reason": "second look", "actor": "real-test"}).json()
    assert o["effective_priority"] == "unsure" and o["priority"] == "raised"
    rv = c.post(f"/studies/{s['study_id']}/review", json={"action": "mark_reviewed", "actor": "real-test"}).json()
    assert rv["status"] == "reviewed" and rv["rank"] is None
    acts = [a["action"] for a in c.get(f"/audit?q={s['study_id']}").json()]
    assert acts == ["mark_reviewed", "override", "scoring", "upload"]


def test_rotated_image_is_flagged_by_the_trained_detector_and_forced_unsure(real_client):
    c = real_client
    rec = next(r for r in demo_studies() if "rotated" in r["study_id"])
    j = c.post("/ood-check", files={"file": ("r.png", image_bytes(rec), "image/png")}).json()
    assert j["ood_flag"] is True and j["source"] == "real" and "trained detector" in j["method"] and j["score"] > j["score_threshold"]
    s = wait_done(c, c.post("/studies", files={"file": ("r.png", image_bytes(rec), "image/png")}).json()["study_id"])
    assert s["ood_flag"] is True and s["priority"] == "unsure" and s["findings"] == [] and s["heatmap_path"] is None
    assert c.post(f"/studies/{s['study_id']}/override", json={"priority": "normal", "reason": "x"}).status_code == 409
    ok = next(r for r in demo_studies() if r["category"] == "queue_nih")
    assert c.post("/ood-check", files={"file": ("ok.png", image_bytes(ok), "image/png")}).json()["ood_flag"] is False


def test_real_numbers_behind_threshold_and_queue_race(real_client):
    c = real_client
    t = c.get("/threshold").json()
    assert t["value"] == 0.4 and t["data_is_real"] is True and abs(t["score_cutoff"] - 0.04765) < 1e-6
    p = c.post("/threshold/preview", json={"value": 0.4}).json()
    assert p["n"] == 11341 and p["n_positive"] == 2307 and 0.90 <= p["sensitivity"] < 0.91 and 0.50 < p["false_alarm_rate"] < 0.53
    assert p["held_out_test"]["n"] == 21845 and p["held_out_test"]["n_positive"] == 4659
    assert c.put("/threshold", json={"value": 0.8, "confirm": True}).status_code == 400, "A's 20% budget does not meet the 90% floor"
    j = c.post("/simulations/queue-race", json={"arrivals_per_hour": 20, "reads_per_hour": 18, "urgent_share": 0.15, "hours": 8, "seed": 1}).json()
    assert j["data_is_real"] is True and "real saved model outputs" in j["assumptions"]["note"]
    assert j["voxelsort"]["urgent_wait_min"] < j["fcfs"]["urgent_wait_min"]
    assert j["voxelsort"]["wrong_flags"] > 0, "the model's wrong flags are counted, not hidden"


def test_seeded_demo_queue_uses_saved_real_outputs_and_real_heatmaps(real_client):
    items = real_client.get("/studies").json()["items"]
    seeded = [i for i in items if i["model_source"] == "saved_demo"]
    assert len(seeded) == 12 and {i["effective_priority"] for i in seeded} == {"raised", "unsure", "normal"}
    by_id = {r["study_id"]: r for r in demo_studies()}
    for i in seeded:
        assert {f["name"]: f["probability"] for f in i["findings"]} == by_id[i["study_id"]]["findings"]
        assert real_client.get(f"/studies/{i['study_id']}/image").status_code == 200
        if not i["ood_flag"]:
            assert real_client.get(f"/studies/{i['study_id']}/heatmap").status_code == 200


def test_site_readiness_with_real_images(real_client):
    """59 demo images scored by the real model: a drift score from A's method, sensitivity from the labelled ones."""
    c = real_client
    recs = demo_studies()
    files = [("files", (f"{r['study_id']}.png", image_bytes(r), "image/png")) for r in recs]
    labelled = [r for r in recs if r["label_any_urgent"] is not None and "__" not in r["study_id"]]
    csv_text = "filename,label\n" + "".join(f"{r['study_id']}.png,{r['label_any_urgent']}\n" for r in labelled)
    job = c.post("/site-readiness", files=files + [("labels", ("labels.csv", csv_text.encode(), "text/csv"))], data={"actor": "real-test"}).json()["job_id"]
    for _ in range(600):
        res = c.get(f"/site-readiness/{job}").json()
        if res["status"] == "done":
            break
    assert res["status"] == "done" and res["model_mode"] == "real" and res["n_images"] == len(recs)
    assert res["drift"]["domain_auroc"] is not None and 0.0 <= res["drift"]["domain_auroc"] <= 1.0
    assert res["drift"]["ood_rate"] > 0.1, "the bundle contains deliberately broken images"
    n_pos = sum(r["label_any_urgent"] for r in labelled)
    assert res["sensitivity"]["n_positive"] == n_pos and res["sensitivity"]["method"] == "clopper-pearson"
    assert res["verdict"] != "ready", "a cherry-picked demo bundle with broken images must never come out as ready"
