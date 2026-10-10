import React from "react";
import { useNavigate } from "react-router-dom";
import {
  AlertCircle, AlertTriangle, ArrowRight, CheckCircle2, ChevronDown, Clock3, FileSearch, Filter, Layers, Plus, RefreshCw, Search, X,
} from "lucide-react";
import { api } from "../services/api.js";
import { usePoll, useNow } from "../lib/hooks.js";
import { cap, formatWait, getActor, pct } from "../lib/format.js";
import { Badge, Banner, Loading, Notice, PriorityBadge, ReasonChip, SectionHeading, StatusBadge } from "../components/ui.jsx";

function QueueControls({ mode, maxWait, onChanged }) {
  const [draft, setDraft] = React.useState(String(maxWait));
  const [msg, setMsg] = React.useState(null);
  const [busy, setBusy] = React.useState(false);
  React.useEffect(() => setDraft(String(maxWait)), [maxWait]);

  const save = async (next) => {
    setBusy(true); setMsg(null);
    try {
      await api.putQueueSettings({ mode: next.mode, max_wait_minutes: Number(next.max_wait_minutes), actor: getActor() || undefined });
      setMsg({ tone: "success", text: next.mode === "fcfs" ? "Fallback on: scans are now read in arrival order." : "Settings saved." });
      onChanged();
    } catch (e) { setMsg({ tone: "warning", text: e.message }); }
    setBusy(false);
  };

  return (
    <section className="panel queue-controls">
      <div className="panel-header">
        <div><h2>Queue controls</h2><p>The fallback switch and the maximum wait cap. Every change is logged.</p></div>
        <Badge tone={mode === "fcfs" ? "warning" : "success"}>{mode === "fcfs" ? "Fallback: first come, first served" : "VoxelSort priority order"}</Badge>
      </div>
      <div className="qc-grid">
        <div>
          <span className="field-label">QUEUE ORDER</span>
          <div className="segmented">
            <button className={mode === "model" ? "selected" : ""} disabled={busy} onClick={() => mode !== "model" && save({ mode: "model", max_wait_minutes: draft })}>VoxelSort priority</button>
            <button className={mode === "fcfs" ? "selected" : ""} disabled={busy} onClick={() => mode !== "fcfs" && save({ mode: "fcfs", max_wait_minutes: draft })}>First come, first served</button>
          </div>
        </div>
        <label className="qc-field">
          <span className="field-label">MAXIMUM WAIT (MINUTES)</span>
          <div className="qc-input">
            <input type="number" min="10" max="1440" value={draft} onChange={(e) => setDraft(e.target.value)} />
            <button className="btn btn-secondary btn-sm" disabled={busy || Number(draft) === maxWait} onClick={() => save({ mode, max_wait_minutes: draft })}>Apply</button>
          </div>
        </label>
      </div>
      <p className="subtle-help">After the maximum wait, a scan jumps ahead of newer scans whatever its priority, so false alarms cannot starve routine patients. A clinician's urgent request always goes first.</p>
      {msg && <Notice tone={msg.tone}>{msg.text}</Notice>}
    </section>
  );
}

export default function Worklist() {
  const navigate = useNavigate();
  const studies = usePoll(() => api.getStudies(), 3000);
  const now = useNow(1000);
  const [query, setQuery] = React.useState("");
  const [priority, setPriority] = React.useState("all");
  const [status, setStatus] = React.useState("all");
  const [sort, setSort] = React.useState("queue");

  if (!studies.data) return <Loading error={studies.error} what="worklist" />;
  const { items, mode, max_wait_minutes: maxWait } = studies.data;

  const liveWait = (c) => (c.status === "reviewed" ? c.wait_seconds : c.wait_seconds + (now - studies.at) / 1000);
  const q = query.trim().toLowerCase();
  const filtered = items
    .filter((c) => !q || c.study_id.toLowerCase().includes(q) || c.findings.some((f) => f.name.includes(q)))
    .filter((c) => priority === "all" || c.effective_priority === priority)
    .filter((c) => status === "all" || c.status === status);
  if (sort === "wait") filtered.sort((a, b) => liveWait(b) - liveWait(a));
  if (sort === "id") filtered.sort((a, b) => a.study_id.localeCompare(b.study_id));

  const open = items.filter((c) => c.status !== "reviewed");
  const oldest = open.length ? Math.max(...open.map(liveWait)) : 0;
  const stats = {
    waiting: open.length,
    raised: open.filter((c) => c.effective_priority === "raised").length,
    unsure: open.filter((c) => c.effective_priority === "unsure").length,
    reviewed: items.length - open.length,
  };

  return (
    <>
      <SectionHeading
        eyebrow="READING QUEUE"
        title="Worklist"
        description="Scans are ranked so likely-urgent ones are read sooner. The model can only move a scan up, never remove it."
        action={<button className="btn btn-primary" onClick={() => navigate("/upload")}><Plus size={17} /> Upload study</button>}
      />
      {studies.error && <Notice>{studies.error}</Notice>}

      <div className="stat-grid">
        <div className="stat-card"><div className="stat-top"><span>Waiting to be read</span><div className="stat-icon blue"><Layers size={17} /></div></div><div className="stat-number">{stats.waiting}<span className="stat-unit">scans</span></div><div className="stat-foot"><span className="tiny-dot blue-dot" /> Oldest waiting {formatWait(oldest)}</div></div>
        <div className="stat-card"><div className="stat-top"><span>Urgent review</span><div className="stat-icon red"><AlertTriangle size={17} /></div></div><div className="stat-number">{stats.raised}<span className="stat-unit">scans</span></div><div className="stat-foot"><span className="tiny-dot red-dot" /> Moved to the top of the queue</div></div>
        <div className="stat-card"><div className="stat-top"><span>AI unsure</span><div className="stat-icon amber"><AlertCircle size={17} /></div></div><div className="stat-number">{stats.unsure}<span className="stat-unit">scans</span></div><div className="stat-foot"><span className="tiny-dot amber-dot" /> Never treated as low risk</div></div>
        <div className="stat-card"><div className="stat-top"><span>Reviewed</span><div className="stat-icon teal"><CheckCircle2 size={17} /></div></div><div className="stat-number">{stats.reviewed}<span className="stat-unit">scans</span></div><div className="stat-foot"><span className="tiny-dot teal-dot" /> Read by a clinician</div></div>
      </div>

      <QueueControls mode={mode} maxWait={maxWait} onChanged={studies.reload} />

      <section className="panel worklist-panel">
        <div className="panel-header">
          <div><h2>Reading queue</h2><p>{mode === "fcfs" ? "First come, first served (fallback is on)." : "Ranked: clinician request, then wait cap, then urgent, AI unsure, routine."}</p></div>
          <button className="btn btn-secondary btn-sm" onClick={() => { setQuery(""); setPriority("all"); setStatus("all"); setSort("queue"); }}><RefreshCw size={15} /> Reset</button>
        </div>
        <div className="table-toolbar">
          <label className="search-box"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search study ID or finding..." />{query && <button onClick={() => setQuery("")} aria-label="Clear search"><X size={15} /></button>}</label>
          <div className="filter-group">
            <label className="select-wrap"><Filter size={15} /><select value={priority} onChange={(e) => setPriority(e.target.value)}><option value="all">All priorities</option><option value="raised">Urgent review</option><option value="unsure">AI unsure</option><option value="normal">Routine</option></select><ChevronDown size={14} /></label>
            <label className="select-wrap"><select value={status} onChange={(e) => setStatus(e.target.value)}><option value="all">All statuses</option><option value="processing">Processing</option><option value="awaiting_review">Awaiting review</option><option value="needs_review">Needs review</option><option value="reviewed">Reviewed</option></select><ChevronDown size={14} /></label>
            <label className="select-wrap"><select value={sort} onChange={(e) => setSort(e.target.value)}><option value="queue">Queue order</option><option value="wait">Longest wait</option><option value="id">Study ID</option></select><ChevronDown size={14} /></label>
          </div>
        </div>
        <div className="table-wrap">
          <table className="study-table">
            <thead><tr><th>#</th><th>STUDY</th><th>TOP FINDING</th><th>PRIORITY</th><th>CONFIDENCE</th><th>WAITING</th><th>STATUS</th><th /></tr></thead>
            <tbody>
              {filtered.map((c) => {
                const w = liveWait(c);
                const overCap = c.status !== "reviewed" && w / 60 >= maxWait;
                return (
                  <tr key={c.study_id} className="clickable-row" onClick={() => navigate(`/study/${c.study_id}`)}>
                    <td className="rank-cell">{c.rank ?? "—"}</td>
                    <td><div className="study-id">{c.study_id}</div><div className="table-sub">Chest X-ray</div></td>
                    <td>
                      {c.findings[0] ? <><div className="finding-main">{cap(c.findings[0].name)} · {pct(c.findings[0].probability)}</div>{c.findings.length > 1 && <div className="table-sub">+{c.findings.length - 1} more</div>}</> : <span className="table-sub">Processing...</span>}
                    </td>
                    <td><PriorityBadge priority={c.effective_priority} /><ReasonChip reason={c.priority_reason} />{c.ood_flag && <div className="table-sub uncertainty-text"><AlertCircle size={12} /> Unlike training data</div>}</td>
                    <td><span className="confidence-band">{c.confidence_band}</span></td>
                    <td><span className={`uploaded-time ${overCap ? "over-cap" : ""}`}><Clock3 size={14} />{c.status === "reviewed" ? "Done" : formatWait(w)}</span>{overCap && <div className="table-sub over-cap">Over {maxWait} min cap</div>}</td>
                    <td><StatusBadge status={c.status} /></td>
                    <td><button className="row-open" aria-label={`Open ${c.study_id}`}><ArrowRight size={17} /></button></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          {filtered.length === 0 && <div className="empty-state"><FileSearch size={28} /><strong>No scans found</strong><span>Try changing your search or filters.</span></div>}
        </div>
        <div className="table-footer"><span>Showing <strong>{filtered.length}</strong> of <strong>{items.length}</strong> scans</span><span className="synthetic-note">Updates every few seconds</span></div>
      </section>

      <Banner icon={AlertCircle} title="Decision support only">
        VoxelSort reorders the reading queue. It does not diagnose, and it never clears a scan. A clinician reads every scan.
      </Banner>
    </>
  );
}
