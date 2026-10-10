import React from "react";
import { Info, X } from "lucide-react";
import { PRIORITY, REASON, STATUS } from "../lib/format.js";

export function Badge({ children, tone = "neutral", dot = false }) {
  return <span className={`badge ${tone}`}>{dot && <i className="badge-dot" />}{children}</span>;
}

export function SectionHeading({ eyebrow, title, description, action }) {
  return (
    <div className="section-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h1>{title}</h1>
        {description && <p>{description}</p>}
      </div>
      {action}
    </div>
  );
}

export function PriorityBadge({ priority }) {
  const p = PRIORITY[priority];
  if (!p) return <Badge tone="neutral">Pending</Badge>;
  return <Badge tone={p.tone} dot>{p.label}</Badge>;
}

export function StatusBadge({ status }) {
  const s = STATUS[status] || { label: status, tone: "neutral" };
  return <Badge tone={s.tone}>{s.label}</Badge>;
}

export function ReasonChip({ reason }) {
  if (!REASON[reason]) return null;
  return <div className="reason-chip">{REASON[reason]}</div>;
}

export function Banner({ icon: Icon = Info, title, children, action }) {
  return (
    <div className="safety-banner">
      <div className="safety-banner-icon"><Icon size={19} /></div>
      <div><strong>{title}</strong><p>{children}</p></div>
      {action}
    </div>
  );
}

export function Notice({ tone = "warning", children }) {
  return <div className={`inline-message ${tone === "success" ? "success-message" : "warning"}`}><Info size={16} />{children}</div>;
}

export function Modal({ onClose, children }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(e) => e.stopPropagation()}>
        <button className="modal-close icon-btn" onClick={onClose} aria-label="Close"><X size={18} /></button>
        {children}
      </div>
    </div>
  );
}

export function Loading({ error, what = "data" }) {
  if (error) return <div className="inline-message warning"><Info size={16} />{error}</div>;
  return <div className="empty-state"><span>Loading {what}...</span></div>;
}
