import React from "react";
import { FileSearch, Info, Upload } from "lucide-react";
import { api, isLive } from "../services/api.js";
import { usePoll } from "../lib/hooks.js";
import { Badge, Banner, Loading, Notice, SectionHeading } from "../components/ui.jsx";

const TABS = ["Calibration", "Cross-dataset", "Per finding", "Shortcut audit", "Where it fails", "OOD check", "Limitations"];

const fmt = (v) => (typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(3)) : v === null || v === undefined ? "—" : String(v));
const label = (k) => k.replace(/_/g, " ");

function NotProvided({ who }) {
  return (
    <div className="empty-state">
      <FileSearch size={28} />
      <strong>Not available yet</strong>
      <span>{who} has not supplied this result. Nothing is shown rather than a made-up number.</span>
    </div>
  );
}

function Table({ rows }) {
  if (!Array.isArray(rows) || rows.length === 0) return null;
  const cols = Object.keys(rows[0]);
  return (
    <div className="table-wrap">
      <table className="study-table">
        <thead><tr>{cols.map((c) => <th key={c}>{label(c).toUpperCase()}</th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c}>{fmt(r[c])}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

function Reliability({ cal }) {
  const W = 320, H = 320, P = 34;
  const x = (v) => P + v * (W - 2 * P), y = (v) => H - P - v * (H - 2 * P);
  const line = (pts) => pts.map((p, i) => `${i ? "L" : "M"}${x(p.confidence)} ${y(p.accuracy)}`).join(" ");
  const Panel = ({ title, pts, color, ece }) => (
    <div className="reliability">
      <h3>{title}{ece !== undefined && <span> · ECE {fmt(ece)}</span>}</h3>
      <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={`${title} reliability diagram`}>
        <rect x={P} y={P} width={W - 2 * P} height={H - 2 * P} fill="none" stroke="#c9d6e3" />
        <path d={`M${x(0)} ${y(0)} L${x(1)} ${y(1)}`} stroke="#8aa0b6" strokeDasharray="5 5" fill="none" />
        <path d={line(pts)} stroke={color} strokeWidth="2.5" fill="none" />
        {pts.map((p, i) => <circle key={i} cx={x(p.confidence)} cy={y(p.accuracy)} r="3.5" fill={color} />)}
        <text x={W / 2} y={H - 8} textAnchor="middle" fontSize="11" fill="#5b6f84">Predicted probability</text>
        <text x="12" y={H / 2} fontSize="11" fill="#5b6f84" transform={`rotate(-90 12 ${H / 2})`} textAnchor="middle">Observed frequency</text>
      </svg>
    </div>
  );
  return (
    <div className="reliability-grid">
      <Panel title="Before calibration" pts={cal.before || []} color="#c2410c" ece={cal.ece_before} />
      <Panel title="After calibration" pts={cal.after || []} color="#0d9488" ece={cal.ece_after} />
      <p className="subtle-help">A well-calibrated model sits on the dashed line: when it says 80%, about 80% are truly positive.</p>
    </div>
  );
}

function OodCheck() {
  const [res, setRes] = React.useState(null);
  const [error, setError] = React.useState("");
  const ref = React.useRef(null);
  const run = async (file) => {
    setRes(null); setError("");
    try { setRes(await api.oodCheck(file)); } catch (e) { setError(e.message); }
  };
  return (
    <section className="panel evidence-text-panel">
      <h2>Out-of-distribution check</h2>
      <p>Try a rotated image, a non-chest image, or a scan from another dataset. The system should say "unsure", never "routine". Show one catch and one miss honestly.</p>
      <button className="btn btn-secondary" onClick={() => ref.current?.click()}><Upload size={16} /> Choose an image</button>
      <input ref={ref} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && run(e.target.files[0])} />
      {error && <Notice>{error}</Notice>}
      {res && (
        <div className="helper-result">
          <div><span>Out of distribution</span><strong>{res.ood_flag ? "Yes: unsure" : "No"}</strong></div>
          <div><span>Score</span><strong>{fmt(res.score)}</strong></div>
          <p>{res.reason}</p>
        </div>
      )}
      {!isLive && <p className="subtle-help">Demo mode cannot run the real detector.</p>}
    </section>
  );
}

export default function Evidence() {
  const [tab, setTab] = React.useState("Calibration");
  const ev = usePoll(() => api.getEvidence(), 0);
  if (ev.loading) return <Loading error={ev.error} what="evidence" />;
  const d = ev.data || {};

  return (
    <>
      <SectionHeading
        eyebrow="TRANSPARENCY"
        title="Evidence"
        description="What the model has been shown to do, where it fails, and what is still unknown. Only real results are shown here."
        action={<Badge tone={ev.data ? "success" : "warning"}>{ev.data ? "Results loaded" : "No results yet"}</Badge>}
      />
      {ev.error && <Notice>{ev.error}</Notice>}
      <div className="evidence-tabs">{TABS.map((t) => <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>{t}</button>)}</div>

      {tab === "Calibration" && (
        <section className="panel evidence-text-panel">
          <h2>Calibration, before and after</h2>
          {d.calibration ? <Reliability cal={d.calibration} /> : <NotProvided who="The ML team" />}
        </section>
      )}
      {tab === "Cross-dataset" && (
        <section className="panel evidence-text-panel">
          <h2>Trained on one dataset, tested on another</h2>
          {d.cross_dataset ? <Table rows={d.cross_dataset} /> : <NotProvided who="The ML team" />}
        </section>
      )}
      {tab === "Per finding" && (
        <section className="panel evidence-text-panel">
          <h2>Performance per finding</h2>
          {d.per_finding ? <Table rows={d.per_finding} /> : <NotProvided who="The ML team" />}
        </section>
      )}
      {tab === "Shortcut audit" && (
        <section className="panel evidence-text-panel">
          <h2>Shortcut audit</h2>
          <p>Checks whether the model relies on things that are not the lungs, such as image borders, markers or text.</p>
          {d.shortcut_audit ? (Array.isArray(d.shortcut_audit) ? <Table rows={d.shortcut_audit} /> : Object.entries(d.shortcut_audit).map(([k, v]) => <div className="evidence-detail-row" key={k}><span>{label(k)}</span><strong>{fmt(v)}</strong></div>)) : <NotProvided who="The evaluation team" />}
        </section>
      )}
      {tab === "Where it fails" && (
        <section className="panel evidence-text-panel">
          <h2>Where it fails</h2>
          {Array.isArray(d.failures) && d.failures.length > 0 ? (
            <div className="failure-gallery">
              {d.failures.map((f, i) => (
                <figure key={i}>{f.image_url && <img src={f.image_url} alt={f.caption || `Failure ${i + 1}`} />}<figcaption><strong>{f.study_id}</strong> {f.caption}</figcaption></figure>
              ))}
            </div>
          ) : <NotProvided who="The ML team" />}
        </section>
      )}
      {tab === "OOD check" && <OodCheck />}
      {tab === "Limitations" && (
        <section className="panel evidence-text-panel">
          <h2>Known limitations</h2>
          <ul className="limitation-list">
            <li>VoxelSort is decision support only. A clinician reads every scan.</li>
            <li>The urgency mapping (which findings count as urgent) is an assumption, not a clinical standard.</li>
            <li>A drift score is a warning signal, not a proven predictor of performance at a new site.</li>
            <li>Queue Race is a simulation under stated assumptions. It is not clinical proof.</li>
            <li>Sensitivity intervals are wide when few urgent scans are available. Check the interval width.</li>
            <li>The two public datasets use different labels, so only the findings they share are compared.</li>
          </ul>
        </section>
      )}
      <Banner icon={Info} title="Be precise about evidence">Only claim what a documented experiment supports. If a result is missing, this page says so.</Banner>
    </>
  );
}
