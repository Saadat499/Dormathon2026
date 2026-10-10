import React from "react";
import { AlertTriangle, CheckCircle2, FileImage, FileText, Info, Upload, XCircle } from "lucide-react";
import { api } from "../services/api.js";
import { getActor, pct } from "../lib/format.js";
import { Badge, Banner, Notice, SectionHeading } from "../components/ui.jsx";

const VERDICT = {
  ready: { label: "Ready", tone: "success", icon: CheckCircle2, text: "The lower end of the interval meets the safety target and no drift warning was raised." },
  use_with_caution: { label: "Use with caution", tone: "warning", icon: AlertTriangle, text: "Sensitivity looks sufficient, but the new site's images drift from the training data. Watch closely." },
  not_ready: { label: "Not ready", tone: "urgent", icon: XCircle, text: "Even the top of the interval is below the safety target." },
  not_enough_evidence: { label: "Not enough evidence", tone: "pending", icon: Info, text: "The sample is too small, or the lower end of the interval is below the safety target, so the target cannot be confirmed yet." },
};

function SampleSizeHelper() {
  const [p, setP] = React.useState({ target_sensitivity: 0.9, expected_sensitivity: 0.95, confidence: 0.95, prevalence: 0.2 });
  const [res, setRes] = React.useState(null);
  const [error, setError] = React.useState("");
  React.useEffect(() => {
    const t = setTimeout(() => api.sampleSize(p).then((r) => { setRes(r); setError(""); }).catch((e) => setError(e.message)), 200);
    return () => clearTimeout(t);
  }, [p]);
  const set = (k) => (e) => setP((old) => ({ ...old, [k]: Number(e.target.value) }));
  return (
    <section className="panel helper-panel">
      <div className="panel-header"><div><h2>Sample-size helper</h2><p>How many scans do you need to confirm a target? Calculated live.</p></div></div>
      <div className="race-grid">
        <label className="form-label">Target sensitivity<input type="number" min="0.5" max="0.99" step="0.01" value={p.target_sensitivity} onChange={set("target_sensitivity")} /></label>
        <label className="form-label">Sensitivity you expect<input type="number" min="0.5" max="0.999" step="0.01" value={p.expected_sensitivity} onChange={set("expected_sensitivity")} /></label>
        <label className="form-label">Confidence<select value={p.confidence} onChange={set("confidence")}><option value="0.9">90%</option><option value="0.95">95%</option><option value="0.99">99%</option></select></label>
        <label className="form-label">Share of scans that are urgent<input type="number" min="0.01" max="0.9" step="0.01" value={p.prevalence} onChange={set("prevalence")} /></label>
      </div>
      {error && <Notice>{error}</Notice>}
      {res && (
        <div className="helper-result">
          <div><span>Urgent scans needed</span><strong>{res.positives_needed ?? "—"}</strong></div>
          <div><span>Total scans needed</span><strong>{res.scans_needed ?? "—"}</strong></div>
          <p>{res.note}</p>
        </div>
      )}
    </section>
  );
}

function IntervalBar({ s, target }) {
  const pos = (v) => `${v * 100}%`;
  return (
    <div className="interval">
      <div className="interval-track">
        <i className="interval-band" style={{ left: pos(s.ci_low), width: pos(s.ci_high - s.ci_low) }} />
        <b className="interval-point" style={{ left: pos(s.point) }} />
        <b className="target-mark" style={{ left: pos(target) }} title="Safety target" />
      </div>
      <div className="interval-scale"><span>0%</span><span>Target {pct(target)}</span><span>100%</span></div>
    </div>
  );
}

export default function SiteReadiness() {
  const [files, setFiles] = React.useState([]);
  const [labels, setLabels] = React.useState(null);
  const [job, setJob] = React.useState(null);
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState("");
  const [busy, setBusy] = React.useState(false);
  const imgRef = React.useRef(null);
  const csvRef = React.useRef(null);

  React.useEffect(() => {
    if (!job || result?.status === "done") return undefined;
    const t = setInterval(() => {
      api.getSiteReadiness(job).then((r) => { if (r.status === "done") setResult(r); }).catch((e) => setError(e.message));
    }, 1500);
    return () => clearInterval(t);
  }, [job, result]);

  const start = async () => {
    setBusy(true); setError(""); setResult(null);
    try { const r = await api.startSiteReadiness(files, labels, getActor() || undefined); setJob(r.job_id); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };
  const reset = () => { setFiles([]); setLabels(null); setJob(null); setResult(null); setError(""); };

  const running = job && !result;
  const v = result && VERDICT[result.verdict];

  return (
    <>
      <SectionHeading
        eyebrow="BEFORE YOU TRUST IT"
        title="Site readiness"
        description="Is it safe to use VoxelSort at a new site? Upload a sample of that site's scans and, if you have them, the true labels."
        action={result && <button className="btn btn-secondary" onClick={reset}>Start again</button>}
      />
      {error && <Notice>{error}</Notice>}

      {!job && (
        <div className="wizard">
          <section className="panel wizard-step">
            <div className="wizard-num">1</div>
            <h2>Sample of the site's scans</h2>
            <p>De-identified images from the new site. More scans give a more reliable drift check.</p>
            <button className="btn btn-secondary" onClick={() => imgRef.current?.click()}><FileImage size={16} /> Choose images</button>
            <input ref={imgRef} type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" multiple hidden onChange={(e) => setFiles(Array.from(e.target.files || []))} />
            {files.length > 0 && <div className="wizard-file"><CheckCircle2 size={16} /> {files.length} image{files.length === 1 ? "" : "s"} selected</div>}
          </section>
          <section className="panel wizard-step">
            <div className="wizard-num">2</div>
            <h2>True labels (optional)</h2>
            <p>A CSV with a <code>label</code> column: 1 for an urgent finding, 0 otherwise. Without it you get the drift check only, not sensitivity.</p>
            <button className="btn btn-secondary" onClick={() => csvRef.current?.click()}><FileText size={16} /> Choose CSV</button>
            <input ref={csvRef} type="file" accept=".csv" hidden onChange={(e) => setLabels(e.target.files?.[0] || null)} />
            {labels && <div className="wizard-file"><CheckCircle2 size={16} /> {labels.name}</div>}
          </section>
          <section className="panel wizard-step">
            <div className="wizard-num">3</div>
            <h2>Run the check</h2>
            <p>Runs in the background and takes a few seconds.</p>
            <button className="btn btn-primary" disabled={busy || files.length === 0} onClick={start}><Upload size={16} /> {busy ? "Starting..." : "Run site check"}</button>
          </section>
        </div>
      )}

      {running && <div className="panel wizard-running"><strong>Checking the sample...</strong><span>Comparing the images with the training data and measuring sensitivity.</span></div>}

      {result && (
        <>
          <section className={`panel verdict-card ${v.tone}`}>
            <v.icon size={28} />
            <div><span className="field-label">VERDICT</span><h2>{v.label}</h2><p>{v.text}</p></div>
          </section>
          <div className="evidence-grid">
            <section className="panel evidence-detail">
              <h2>Drift check</h2>
              <div className="evidence-detail-row"><span>Drift score</span><strong>{result.drift.score}</strong></div>
              <div className="evidence-detail-row"><span>Warning</span><Badge tone={result.drift.warning ? "warning" : "success"}>{result.drift.warning ? "Raised" : "None"}</Badge></div>
              <p className="subtle-help">{result.drift.note}</p>
            </section>
            <section className="panel evidence-detail">
              <h2>Sensitivity at this site</h2>
              {result.sensitivity ? (
                <>
                  <div className="evidence-detail-row"><span>Measured</span><strong>{pct(result.sensitivity.point, 1)} ({result.sensitivity.n_positive} urgent scans)</strong></div>
                  <div className="evidence-detail-row"><span>Interval ({result.sensitivity.method})</span><strong>{pct(result.sensitivity.ci_low, 1)} to {pct(result.sensitivity.ci_high, 1)}</strong></div>
                  <div className="evidence-detail-row"><span>Interval width</span><strong>{pct(result.sensitivity.width, 1)}</strong></div>
                  <IntervalBar s={result.sensitivity} target={result.target} />
                </>
              ) : <p className="subtle-help">{result.note || "No labels were provided, so sensitivity could not be measured."}</p>}
            </section>
          </div>
          {result.note && result.sensitivity && <Notice>{result.note}</Notice>}
        </>
      )}

      <SampleSizeHelper />
      <Banner icon={Info} title="A readiness check is not an approval">
        It tells you whether the evidence supports using VoxelSort at this site. Governance, safety review and any regulatory assessment remain separate. A clinician reads every scan.
      </Banner>
    </>
  );
}
