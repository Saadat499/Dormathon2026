// The ONLY place the frontend talks to a backend. Shapes follow docs/API_CONTRACT.md.
//
//   VITE_API_URL set   -> live backend (for example https://chesttriage-api.onrender.com)
//   VITE_API_URL empty -> demo backend in the browser (services/mock.js), clearly labelled in the UI
import { mockApi } from "./mock.js";

const BASE = (import.meta.env.VITE_API_URL || "").replace(/\/$/, "");
export const isLive = BASE !== "";

async function req(path, { method = "GET", json, form } = {}) {
  let res;
  try {
    res = await fetch(BASE + path, {
      method,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : form,
    });
  } catch {
    throw new Error("Cannot reach the backend. It may be starting up (free hosting sleeps when idle). Try again in a minute.");
  }
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* keep statusText */ }
    const err = new Error(detail || `Request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}
const qs = (obj) => {
  const p = new URLSearchParams();
  Object.entries(obj || {}).forEach(([k, v]) => { if (v !== undefined && v !== null && v !== "") p.set(k, v); });
  const s = p.toString();
  return s ? `?${s}` : "";
};
const enc = encodeURIComponent;

const liveApi = {
  mode: "live",
  getStudies: () => req("/studies"),
  getStudy: (id) => req(`/studies/${enc(id)}`),
  uploadStudy(file, clinicianUrgent, actor) {
    const form = new FormData();
    form.append("file", file);
    form.append("clinician_urgent", clinicianUrgent ? "true" : "false");
    if (actor) form.append("actor", actor);
    return req("/studies", { method: "POST", form });
  },
  imageUrl: (id) => `${BASE}/studies/${enc(id)}/image`,
  heatmapUrl: (study) => (study.heatmap_path ? `${BASE}/studies/${enc(study.study_id)}/heatmap` : null),
  review: (id, body) => req(`/studies/${enc(id)}/review`, { method: "POST", json: body }),
  urgentRequest: (id, body) => req(`/studies/${enc(id)}/urgent-request`, { method: "POST", json: body }),
  override: (id, body) => req(`/studies/${enc(id)}/override`, { method: "POST", json: body }),
  getQueueSettings: () => req("/queue/settings"),
  putQueueSettings: (body) => req("/queue/settings", { method: "PUT", json: body }),
  getThreshold: () => req("/threshold"),
  previewThreshold: (value) => req("/threshold/preview", { method: "POST", json: { value } }),
  putThreshold: (body) => req("/threshold", { method: "PUT", json: body }),
  runQueueRace: (body) => req("/simulations/queue-race", { method: "POST", json: body }),
  sampleSize: (params) => req(`/site-readiness/sample-size${qs(params)}`),
  startSiteReadiness(files, labelsFile, actor) {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    if (labelsFile) form.append("labels", labelsFile);
    if (actor) form.append("actor", actor);
    return req("/site-readiness", { method: "POST", form });
  },
  getSiteReadiness: (jobId) => req(`/site-readiness/${enc(jobId)}`),
  oodCheck(file) {
    const form = new FormData();
    form.append("file", file);
    return req("/ood-check", { method: "POST", form });
  },
  async getEvidence() {
    try { return await req("/evidence"); } catch (e) { if (e.status === 404) return null; throw e; }
  },
  getAudit: (params) => req(`/audit${qs(params)}`),
  async downloadAuditCsv() {
    const a = document.createElement("a");
    a.href = `${BASE}/audit.csv`;
    a.download = "voxelsort-audit.csv";
    a.click();
  },
};

export const api = isLive ? liveApi : mockApi;
