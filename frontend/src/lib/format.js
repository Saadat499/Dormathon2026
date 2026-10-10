export const cap = (s) => (s ? s.charAt(0).toUpperCase() + s.slice(1) : s);
export const pct = (p, digits = 0) => `${(p * 100).toFixed(digits)}%`;

export function formatWait(totalSeconds) {
  const s = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, "0")}m`;
  return `${m}m ${String(sec).padStart(2, "0")}s`;
}

export const formatTime = (iso) =>
  new Date(iso).toLocaleString([], { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" });

// Display labels. The API only ever uses raised | normal | unsure.
export const PRIORITY = {
  raised: { label: "Urgent review", tone: "urgent" },
  unsure: { label: "AI unsure", tone: "review" },
  normal: { label: "Routine", tone: "routine" },
};
export const STATUS = {
  processing: { label: "Processing", tone: "pending" },
  awaiting_review: { label: "Awaiting review", tone: "pending" },
  needs_review: { label: "Needs review", tone: "pending" },
  reviewed: { label: "Reviewed", tone: "success" },
};
export const REASON = {
  clinician_request: "Clinician request",
  wait_cap: "Wait cap reached",
  coordinator_override: "Coordinator override",
};

// A name the reviewer may type in. It is only recorded in the audit log; there is no login.
const KEY = "voxelsort.actor";
export const getActor = () => { try { return localStorage.getItem(KEY) || ""; } catch { return ""; } };
export const setActor = (v) => { try { localStorage.setItem(KEY, v); } catch { /* ignore */ } };
