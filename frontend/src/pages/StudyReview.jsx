import React from "react";
import { useNavigate, useParams } from "react-router-dom";
import {
  Activity, AlertCircle, AlertTriangle, ArrowLeft, Check, CheckCircle2, Clock3, FileImage, History, Info, Layers, RefreshCw, ShieldCheck, Siren, ZoomIn, ZoomOut,
} from "lucide-react";
import { api, isLive } from "../services/api.js";
import { usePoll } from "../lib/hooks.js";
import { cap, formatTime, getActor, pct } from "../lib/format.js";
import { Badge, Loading, Modal, Notice, PriorityBadge, ReasonChip, StatusBadge } from "../components/ui.jsx";
import DemoXray from "../components/DemoXray.jsx";

function Viewer({ study, view, opacity, zoom }) {
  const [imgError, setImgError] = React.useState(false);
  const heat = api.heatmapUrl(study);
  if (!isLive) return <DemoXray heatmap={view === "heatmap"} opacity={opacity} zoom={zoom} />;
  return (
    <div className="xray-stage">
      {imgError ? (
        <div className="empty-state"><Info size={22} /><strong>Image not available</strong><span>The backend did not return an image for this study.</span></div>
      ) : (
        <div className="img-stack" style={{ transform: `scale(${zoom})` }}>
          <img src={api.imageUrl(study.study_id)} alt={`Chest X-ray ${study.study_id}`} onError={() => setImgError(true)} />
          {view === "heatmap" && heat && <img className="heat-layer" src={heat} alt="Grad-CAM heatmap" style={{ opacity: opacity / 100 }} />}
        </div>
      )}
    </div>
  );
}

export default function StudyReview() {
  const { studyId } = useParams();
  const navigate = useNavigate();
  const [study, setStudy] = React.useState(null);
  const poll = usePoll(async () => { const s = await api.getStudy(studyId); setStudy(s); return s; }, study?.status === "processing" ? 2000 : 8000, [studyId]);
  const [view, setView] = React.useState("original");
  const [opacity, setOpacity] = React.useState(48);
  const [zoom, setZoom] = React.useState(1);
  const [modal, setModal] = React.useState("");
  const [msg, setMsg] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  const [reason, setReason] = React.useState("Model finding does not match image");
  const [humanPriority, setHumanPriority] = React.useState("normal");
  const [notes, setNotes] = React.useState("");
  const [ovPriority, setOvPriority] = React.useState("raised");
  const [ovReason, setOvReason] = React.useState("");

  if (!study) return <Loading error={poll.error} what="study" />;
  const heatReady = !isLive || !!api.heatmapUrl(study);
  const actor = getActor() || undefined;

  const run = async (fn, doneText, after) => {
    setBusy(true); setMsg(null);
    try {
      const s = await fn();
      setStudy(s);
      setMsg({ tone: "success", text: doneText });
      setModal("");
      after?.();
    } catch (e) { setMsg({ tone: "warning", text: e.message }); }
    setBusy(false);
  };

  return (
    <>
      <div className="study-page-heading">
        <button className="back-link" onClick={() => navigate("/")}><ArrowLeft size={16} /> Worklist</button>
        <div className="study-heading-row">
          <div>
            <div className="eyebrow">STUDY REVIEW</div>
            <h1>{study.study_id} <span className="title-divider">/</span> Chest X-ray</h1>
            <div className="study-meta-line">
              <span><Clock3 size={14} /> Uploaded {formatTime(study.uploaded_at)}</span>
              <span><FileImage size={14} /> De-identified study</span>
              {study.rank && <span>Queue position {study.rank}</span>}
              <StatusBadge status={study.status} />
            </div>
          </div>
          <button className="btn btn-secondary" onClick={() => navigate("/audit")}><History size={16} /> Audit log</button>
        </div>
      </div>
      {msg && <Notice tone={msg.tone}>{msg.text}</Notice>}

      <div className="review-layout">
        <section className="panel image-panel">
          <div className="panel-header image-panel-header"><div><h2>Image viewer</h2><p>{isLive ? "Original and Grad-CAM heatmap" : "Illustrative placeholder in demo mode"}</p></div><button className="icon-btn" onClick={() => setZoom(1)} title="Reset zoom"><RefreshCw size={16} /></button></div>
          <div className="viewer-toolbar">
            <div className="segmented">
              <button className={view === "original" ? "selected" : ""} onClick={() => setView("original")}>Original</button>
              <button className={view === "heatmap" ? "selected" : ""} onClick={() => setView("heatmap")}>Heatmap</button>
            </div>
            <div className="zoom-controls">
              <button className="icon-btn" onClick={() => setZoom((z) => Math.max(0.8, z - 0.1))} aria-label="Zoom out"><ZoomOut size={16} /></button>
              <span>{Math.round(zoom * 100)}%</span>
              <button className="icon-btn" onClick={() => setZoom((z) => Math.min(1.4, z + 0.1))} aria-label="Zoom in"><ZoomIn size={16} /></button>
            </div>
          </div>
          <Viewer study={study} view={view} opacity={opacity} zoom={zoom} />
          {view === "heatmap" && (
            <div className="heatmap-controls">
              {heatReady ? (
                <>
                  <div className="heatmap-caption"><span><Layers size={14} /> Heatmap overlay</span><strong>{opacity}% opacity</strong></div>
                  <input aria-label="Heatmap opacity" type="range" min="10" max="90" value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} />
                  <div className="heatmap-legend"><span>Lower activation</span><span><i /> Higher activation</span></div>
                </>
              ) : <p className="subtle-help">No heatmap has been generated for this study yet.</p>}
              <p className="subtle-help">{isLive ? "A heatmap shows where the model looked. It does not prove why it decided." : "Demo overlay only. It is not produced by a model."}</p>
            </div>
          )}
        </section>

        <aside className="review-right">
          <section className="panel triage-panel">
            <div className="panel-header"><div><h2>Triage card</h2><p>{isLive ? "Calibrated model output" : "Synthetic demo output"}</p></div><div className="triage-symbol"><Activity size={18} /></div></div>
            <div className="priority-block">
              <span className="field-label">QUEUE PRIORITY</span>
              <div className="priority-line"><PriorityBadge priority={study.effective_priority} /><span className="confidence-text">Confidence {study.confidence_band}</span></div>
              <ReasonChip reason={study.priority_reason} />
              <p>{study.plain_language}</p>
            </div>
            {study.ood_flag && (
              <div className="uncertainty-callout"><AlertTriangle size={17} /><div><strong>Unlike the training data</strong><p>This image looks different from what the model learned from. It is treated as unsure, never as low risk.</p></div></div>
            )}
            <div className="findings-heading"><h3>Findings</h3><span>Calibrated probability</span></div>
            <div className="findings-list">
              {study.findings.length === 0 && <p className="subtle-help">Waiting for model output...</p>}
              {study.findings.map((f) => (
                <div className="finding-row" key={f.name}>
                  <div className="finding-name">{cap(f.name)}</div>
                  <div className="finding-score"><div className="score-track"><span style={{ width: `${f.probability * 100}%` }} /></div><strong>{pct(f.probability)}</strong></div>
                </div>
              ))}
            </div>
            <div className="triage-disclaimer"><ShieldCheck size={16} /><span>Decision support only. A clinician reads every scan.</span></div>
          </section>

          <section className="panel action-panel">
            <h2>Actions</h2>
            <p>Every action is written to the audit log.</p>
            <button className="btn btn-primary btn-block" disabled={busy || study.status === "reviewed"} onClick={() => run(() => api.review(study.study_id, { action: "mark_reviewed", actor }), "Marked as reviewed.")}><Check size={17} /> Mark as reviewed</button>
            <button className="btn btn-secondary btn-block" disabled={busy || study.status === "reviewed"} onClick={() => setModal("flag")}><AlertCircle size={17} /> Flag disagreement</button>
            <button className="btn btn-secondary btn-block" disabled={busy || study.clinician_urgent || study.status === "reviewed"} onClick={() => run(() => api.urgentRequest(study.study_id, { note: "Clinician requested an urgent read.", actor }), "Urgent read requested. This scan now goes to the top of the queue.")}><Siren size={17} /> {study.clinician_urgent ? "Urgent read requested" : "Clinician requests urgent read"}</button>
            <button className="btn btn-secondary btn-block" disabled={busy || study.status === "reviewed"} onClick={() => setModal("override")}>Coordinator override</button>
            <div className="action-foot"><Info size={14} /> A clinician's urgent request always beats the model.</div>
          </section>
        </aside>
      </div>

      {modal === "flag" && (
        <Modal onClose={() => setModal("")}>
          <div className="modal-icon amber"><AlertTriangle size={20} /></div>
          <h2>Flag disagreement</h2>
          <p>Record where your view differs from the model's.</p>
          <label className="form-label">Reason<select value={reason} onChange={(e) => setReason(e.target.value)}><option>Model finding does not match image</option><option>Priority seems too high</option><option>Priority seems too low</option><option>Image quality concern</option><option>Other / requires further review</option></select></label>
          <label className="form-label">Your priority<select value={humanPriority} onChange={(e) => setHumanPriority(e.target.value)}><option value="raised">Urgent review</option><option value="unsure">AI unsure</option><option value="normal">Routine</option></select></label>
          <label className="form-label">Notes (optional)<textarea rows="3" value={notes} onChange={(e) => setNotes(e.target.value)} placeholder="A short note for the audit log" /></label>
          <div className="modal-actions"><button className="btn btn-secondary" onClick={() => setModal("")}>Cancel</button><button className="btn btn-primary" disabled={busy} onClick={() => run(() => api.review(study.study_id, { action: "flag_disagreement", reason, notes, human_priority: humanPriority, actor }), "Disagreement recorded.")}>Save to audit log</button></div>
        </Modal>
      )}
      {modal === "override" && (
        <Modal onClose={() => setModal("")}>
          <div className="modal-icon amber"><AlertTriangle size={20} /></div>
          <h2>Coordinator override</h2>
          <p>Set this scan's queue priority yourself. A reason is required and is logged.</p>
          <label className="form-label">Priority<select value={ovPriority} onChange={(e) => setOvPriority(e.target.value)}><option value="raised">Urgent review</option><option value="unsure">AI unsure</option><option value="normal">Routine</option></select></label>
          <label className="form-label">Reason (required)<textarea rows="3" value={ovReason} onChange={(e) => setOvReason(e.target.value)} placeholder="Why are you overriding the model?" /></label>
          <div className="modal-actions"><button className="btn btn-secondary" onClick={() => setModal("")}>Cancel</button><button className="btn btn-primary" disabled={busy || !ovReason.trim()} onClick={() => run(() => api.override(study.study_id, { priority: ovPriority, reason: ovReason, actor }), "Override applied.", () => setOvReason(""))}>Apply override</button></div>
        </Modal>
      )}
    </>
  );
}
