import React from "react";
import { Activity, AlertCircle, ArrowRight, Clock3 } from "lucide-react";
import { api } from "../services/api.js";
import { Badge, Banner, Notice, SectionHeading } from "../components/ui.jsx";

const FIELDS = [
  ["arrivals_per_hour", "Scans arriving per hour", 1, 200, 1],
  ["reads_per_hour", "Scans one reader can read per hour", 1, 200, 1],
  ["urgent_share", "Share of scans that are truly urgent", 0.01, 0.9, 0.01],
  ["hours", "Hours to simulate", 1, 24, 1],
  ["seed", "Random seed (same seed, same result)", 1, 9999, 1],
];

const na = (v, suffix = "") => (v === null || v === undefined ? "n/a" : `${v}${suffix}`);

function Column({ title, subtitle, tone, data, order }) {
  return (
    <section className="panel queue-panel">
      <div className="queue-panel-head">
        <div><div className={`queue-panel-icon ${tone}`}>{tone === "blue" ? <Clock3 size={17} /> : <Activity size={17} />}</div><div><h2>{title}</h2><p>{subtitle}</p></div></div>
      </div>
      <div className="race-metrics">
        <div><span>Urgent scans wait</span><strong>{data.urgent_wait_min} min</strong></div>
        <div><span>Extra wait for others</span><strong>{data.other_extra_wait_min > 0 ? "+" : ""}{data.other_extra_wait_min} min</strong></div>
        <div><span>Missed urgent scans</span><strong>{na(data.missed_urgent)}</strong></div>
        <div><span>Wrong urgent flags</span><strong>{na(data.wrong_flags)}</strong></div>
      </div>
      <div className="queue-list">
        {order.map((c, i) => (
          <div className="queue-item" key={c.id}>
            <span className="queue-position">{String(i + 1).padStart(2, "0")}</span>
            <div className="queue-study"><strong>{c.id}</strong><span>waited {c.wait_min} min</span></div>
            <Badge tone={c.urgent ? "urgent" : "routine"}>{c.urgent ? "Truly urgent" : "Routine"}</Badge>
          </div>
        ))}
      </div>
    </section>
  );
}

export default function QueueRace() {
  const [params, setParams] = React.useState({ arrivals_per_hour: 20, reads_per_hour: 18, urgent_share: 0.15, hours: 8, seed: 1 });
  const [result, setResult] = React.useState(null);
  const [error, setError] = React.useState("");
  const [busy, setBusy] = React.useState(false);

  const run = async () => {
    setBusy(true); setError("");
    try { setResult(await api.runQueueRace(Object.fromEntries(Object.entries(params).map(([k, v]) => [k, Number(v)])))); }
    catch (e) { setError(e.message); }
    setBusy(false);
  };

  return (
    <>
      <SectionHeading
        eyebrow="SIMULATED IMPACT"
        title="Queue Race"
        description="The same scans arrive at the same reader twice: once in arrival order, once ranked by VoxelSort. Misses and wrong flags are counted, not hidden."
        action={<button className="btn btn-primary" disabled={busy} onClick={run}><Activity size={16} /> {busy ? "Running..." : "Run the race"}</button>}
      />
      <section className="panel race-form">
        <div className="panel-header"><div><h2>Assumptions</h2><p>Change them and run again. The results state exactly what was assumed.</p></div></div>
        <div className="race-grid">
          {FIELDS.map(([key, label, min, max, step]) => (
            <label key={key} className="form-label">{label}
              <input type="number" min={min} max={max} step={step} value={params[key]} onChange={(e) => setParams((p) => ({ ...p, [key]: e.target.value }))} />
            </label>
          ))}
        </div>
      </section>
      {error && <Notice>{error}</Notice>}

      {result && (
        <>
          <div className="queue-summary">
            <div className="queue-summary-icon blue"><Clock3 size={20} /></div>
            <div><span>First come, first served</span><strong>Urgent scans wait {result.fcfs.urgent_wait_min} min</strong><p>{result.n_cases} scans, {result.n_urgent} truly urgent.</p></div>
            <ArrowRight className="queue-summary-arrow" />
            <div className="queue-summary-icon teal"><Activity size={20} /></div>
            <div><span>VoxelSort ranked</span><strong>Urgent scans wait {result.voxelsort.urgent_wait_min} min</strong><p>Others wait {result.voxelsort.other_extra_wait_min >= 0 ? "+" : ""}{result.voxelsort.other_extra_wait_min} min on average.</p></div>
          </div>
          <div className="queue-columns">
            <Column title="First come, first served" subtitle="Baseline: no model" tone="blue" data={result.fcfs} order={result.read_order.fcfs} />
            <Column title="VoxelSort ranked" subtitle="Includes misses and wrong flags" tone="teal" data={result.voxelsort} order={result.read_order.voxelsort} />
          </div>
          <section className="panel assumptions-panel">
            <h2>Stated assumptions</h2>
            <p>{result.assumptions.note}</p>
            <div className="assumption-list">
              {Object.entries(result.assumptions).filter(([k]) => k !== "note").map(([k, v]) => <span key={k}><em>{k.replace(/_/g, " ")}</em> {String(v)}</span>)}
            </div>
          </section>
        </>
      )}
      <Banner icon={AlertCircle} title="A simulation is not clinical proof">
        This shows what the ranking rule does under the stated assumptions. It does not show that patients would do better. Say so when you present it.
      </Banner>
    </>
  );
}
