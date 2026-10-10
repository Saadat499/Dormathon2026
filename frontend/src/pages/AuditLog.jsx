import React from "react";
import { ChevronDown, Download, History, Info, Search } from "lucide-react";
import { api } from "../services/api.js";
import { usePoll } from "../lib/hooks.js";
import { formatTime } from "../lib/format.js";
import { Badge, Banner, Loading, SectionHeading } from "../components/ui.jsx";

const ACTIONS = ["upload", "mark_reviewed", "flag_disagreement", "urgent_request", "override", "queue_settings", "threshold_change", "site_readiness_run"];
const tone = (a) => (["flag_disagreement", "override", "threshold_change", "queue_settings"].includes(a) ? "warning" : a === "urgent_request" ? "urgent" : "success");

export default function AuditLog() {
  const [query, setQuery] = React.useState("");
  const [action, setAction] = React.useState("");
  const log = usePoll(() => api.getAudit({ action: action || undefined, q: query || undefined }), 5000, [action, query]);
  if (!log.data) return <Loading error={log.error} what="audit log" />;
  const entries = log.data;

  return (
    <>
      <SectionHeading
        eyebrow="ACCOUNTABILITY"
        title="Audit log"
        description="Every action: uploads, reviews, overrides, urgent requests, queue and threshold changes."
        action={<button className="btn btn-secondary" onClick={() => api.downloadAuditCsv()} disabled={!entries.length}><Download size={16} /> Download CSV</button>}
      />
      <div className="audit-summary"><div className="audit-summary-icon"><History size={20} /></div><div><strong>{entries.length} events shown</strong><span>Newest first. Updates every few seconds.</span></div></div>
      <section className="panel audit-panel">
        <div className="table-toolbar">
          <label className="search-box"><Search size={17} /><input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search study ID..." /></label>
          <label className="select-wrap"><select value={action} onChange={(e) => setAction(e.target.value)}><option value="">All actions</option>{ACTIONS.map((a) => <option key={a} value={a}>{a.replace(/_/g, " ")}</option>)}</select><ChevronDown size={14} /></label>
        </div>
        {entries.length === 0 ? (
          <div className="empty-state"><History size={28} /><strong>No events</strong><span>Actions you take will appear here.</span></div>
        ) : (
          <div className="table-wrap">
            <table className="study-table audit-table">
              <thead><tr><th>TIME</th><th>ACTION</th><th>STUDY</th><th>DETAIL</th><th>ACTOR</th></tr></thead>
              <tbody>
                {entries.map((a) => (
                  <tr key={a.id}>
                    <td>{formatTime(a.timestamp)}</td>
                    <td><Badge tone={tone(a.action)}>{a.action.replace(/_/g, " ")}</Badge></td>
                    <td>{a.study_id ? <strong className="study-id">{a.study_id}</strong> : "—"}</td>
                    <td><div className="finding-main">{a.detail}</div></td>
                    <td>{a.actor}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
      <Banner icon={Info} title="Not a tamper-proof record">There is no login, so the actor is whatever name was typed in. This log records what happened; it does not prove who did it.</Banner>
    </>
  );
}
