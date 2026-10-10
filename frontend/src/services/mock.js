// DEMO BACKEND. Used only when VITE_API_URL is not set.
//
// It follows docs/API_CONTRACT.md so the screens can be built and rehearsed before the
// real FastAPI endpoints exist. Everything here runs in the browser and resets on refresh.
// ALL data is synthetic. Nothing here is a model output or a clinical result.
// The queue-ordering rule below mirrors the contract; the real backend owns this rule.

const rng = (seed) => {
  let a = seed >>> 0;
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};
const randn = (r) => {
  let u = 0, v = 0;
  while (!u) u = r();
  while (!v) v = r();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};
const clamp01 = (x) => Math.min(0.999, Math.max(0.001, x));
const sleep = (ms) => new Promise((res) => setTimeout(res, ms));
const FINDINGS = ["pneumothorax", "effusion", "consolidation", "edema", "atelectasis", "cardiomegaly"];
const PRIORITY_ORDER = { raised: 0, unsure: 1, normal: 2 };
const BAND = 0.15; // scores within BAND below the threshold are "unsure"

let settings = { mode: "model", max_wait_minutes: 120 };
let threshold = { value: 0.42, min_sensitivity_target: 0.9 }; // highest value that still meets the target on the saved predictions
let audit = [];
let auditId = 0;

// "Saved predictions": synthetic scores with known labels, used by the threshold preview.
const SAVED = (() => {
  const r = rng(7);
  return Array.from({ length: 600 }, () => {
    const pos = r() < 0.2;
    return { pos, score: clamp01((pos ? 0.72 : 0.22) + randn(r) * (pos ? 0.18 : 0.16)) };
  });
})();

export function wilson(x, n, z) {
  if (n === 0) return [0, 1];
  const p = x / n;
  const d = 1 + (z * z) / n;
  const c = p + (z * z) / (2 * n);
  const h = z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return [Math.max(0, (c - h) / d), Math.min(1, (c + h) / d)];
}
const zFor = (conf) => (conf >= 0.99 ? 2.576 : conf >= 0.95 ? 1.96 : 1.645);

function derivePriority(probs, ood) {
  const top = Math.max(...Object.values(probs));
  if (ood) return "unsure";
  if (top >= threshold.value) return "raised";
  if (top >= threshold.value - BAND) return "unsure";
  return "normal";
}

const now0 = Date.now();
const seed = (id, minsAgo, probs, extra = {}) => ({
  study_id: id, uploaded_ms: now0 - minsAgo * 60000, probs, ood: false,
  clinician_urgent: false, override_priority: null, reviewed_ms: null,
  flagged: false, processing_until: 0, ...extra,
});
let studies = [
  seed("CXR-1042", 6, { pneumothorax: 0.82, effusion: 0.26, atelectasis: 0.11 }),
  seed("CXR-1041", 14, { edema: 0.38, cardiomegaly: 0.31, effusion: 0.12 }),
  seed("CXR-1040", 22, { cardiomegaly: 0.12, effusion: 0.08, edema: 0.04 }),
  seed("CXR-1039", 35, { effusion: 0.77, atelectasis: 0.44, consolidation: 0.2 }),
  seed("CXR-1038", 51, { consolidation: 0.35, effusion: 0.22, edema: 0.1 }, { ood: true }),
  seed("CXR-1037", 78, { atelectasis: 0.09, edema: 0.05, effusion: 0.03 }),
  seed("CXR-1036", 131, { cardiomegaly: 0.18, effusion: 0.07, edema: 0.03 }),
  seed("CXR-1035", 160, { edema: 0.06, effusion: 0.03, atelectasis: 0.02 }, { reviewed_ms: now0 - 20 * 60000 }),
];
let uploadCounter = 1100;

const pct = (p) => Math.round(p * 100);
const cap = (s) => s.charAt(0).toUpperCase() + s.slice(1);

// Fixed templates filled with the real outputs. Nothing here is free-written medical reasoning.
function plainLanguage(s, priority, findings) {
  const top = findings[0];
  if (!top) return "This scan is still being processed. It will be read by a clinician.";
  if (s.ood) {
    return "This image looks different from the scans VoxelSort was built on, so the system is not sure about it. It has been moved up the queue, not down. A clinician will read the scan.";
  }
  if (priority === "raised") {
    return `VoxelSort estimates a ${pct(top.probability)}% likelihood of ${top.name} on this scan, so it was moved up the reading queue. This is a prioritisation aid, not a diagnosis. A clinician will read the scan.`;
  }
  if (priority === "unsure") {
    return `VoxelSort is not sure about this scan (highest estimate: ${pct(top.probability)}% ${top.name}). It was placed ahead of routine scans so a person looks at it sooner. This is not a diagnosis.`;
  }
  return `VoxelSort did not flag an urgent finding on this scan (highest estimate: ${pct(top.probability)}% ${top.name}). It stays in the normal reading order. This is not an all-clear: a clinician will still read the scan.`;
}

function build(s, nowMs) {
  const processing = nowMs < s.processing_until;
  const findings = processing
    ? []
    : Object.entries(s.probs).sort((a, b) => b[1] - a[1]).map(([name, probability]) => ({ name, probability }));
  const priority = processing ? null : derivePriority(s.probs, s.ood);
  const top = findings[0]?.probability ?? 0;
  const effective = s.override_priority || priority || "unsure"; // not yet predicted: never treated as low risk
  const reviewed = s.reviewed_ms != null;
  const waitMs = (reviewed ? s.reviewed_ms : nowMs) - s.uploaded_ms;
  const overCap = !reviewed && waitMs / 60000 >= settings.max_wait_minutes;
  let reason = "model";
  if (s.clinician_urgent) reason = "clinician_request";
  else if (overCap && settings.mode === "model") reason = "wait_cap";
  else if (s.override_priority) reason = "coordinator_override";
  let status = "awaiting_review";
  if (processing) status = "processing";
  else if (reviewed) status = "reviewed";
  else if (effective === "unsure") status = "needs_review";
  return {
    study_id: s.study_id,
    findings,
    priority,
    confidence_band: processing ? "pending" : `${Math.max(0, top - 0.11).toFixed(2)}-${Math.min(1, top + 0.07).toFixed(2)}`,
    ood_flag: !!s.ood,
    heatmap_path: null,
    uploaded_at: new Date(s.uploaded_ms).toISOString(),
    wait_seconds: Math.round(waitMs / 1000),
    rank: null,
    effective_priority: effective,
    priority_reason: reason,
    clinician_urgent: s.clinician_urgent,
    status,
    flagged_disagreement: s.flagged,
    plain_language: plainLanguage(s, priority || "unsure", findings),
  };
}

// The single ranking rule from the contract.
function rankOrder(list) {
  const open = list.filter((s) => s.status !== "reviewed");
  const done = list.filter((s) => s.status === "reviewed").sort((a, b) => Date.parse(b.uploaded_at) - Date.parse(a.uploaded_at));
  const byArrival = (a, b) => Date.parse(a.uploaded_at) - Date.parse(b.uploaded_at);
  if (settings.mode === "fcfs") open.sort(byArrival);
  else {
    open.sort((a, b) => {
      if (a.clinician_urgent !== b.clinician_urgent) return a.clinician_urgent ? -1 : 1;
      const ca = a.priority_reason === "wait_cap", cb = b.priority_reason === "wait_cap";
      if (ca !== cb) return ca ? -1 : 1;
      const pa = PRIORITY_ORDER[a.effective_priority], pb = PRIORITY_ORDER[b.effective_priority];
      if (pa !== pb) return pa - pb;
      return byArrival(a, b);
    });
  }
  open.forEach((s, i) => { s.rank = i + 1; });
  return [...open, ...done];
}

function log(action, study_id, detail, actor) {
  audit.unshift({ id: ++auditId, timestamp: new Date().toISOString(), actor: actor || "unnamed", action, study_id: study_id || null, detail });
}
const find = (id) => {
  const s = studies.find((x) => x.study_id === id);
  if (!s) throw new Error(`Study ${id} not found`);
  return s;
};

export const mockApi = {
  mode: "demo",

  async getStudies() {
    await sleep(120);
    const t = Date.now();
    return { mode: settings.mode, max_wait_minutes: settings.max_wait_minutes, items: rankOrder(studies.map((s) => build(s, t))) };
  },
  async getStudy(id) {
    await sleep(80);
    const t = Date.now();
    const item = rankOrder(studies.map((s) => build(s, t))).find((s) => s.study_id === id);
    if (!item) throw new Error(`Study ${id} not found`);
    return item;
  },
  async uploadStudy(file, clinicianUrgent, actor) {
    await sleep(400);
    uploadCounter += 1;
    const id = `CXR-${uploadCounter}`;
    const r = rng(file.size + file.name.length * 31);
    const lead = FINDINGS[Math.floor(r() * FINDINGS.length)];
    const probs = { [lead]: clamp01(r()) };
    FINDINGS.filter((f) => f !== lead).slice(0, 2).forEach((f) => { probs[f] = clamp01(r() * 0.4); });
    studies.push({
      study_id: id, uploaded_ms: Date.now(), probs, ood: false, clinician_urgent: !!clinicianUrgent,
      override_priority: null, reviewed_ms: null, flagged: false, processing_until: Date.now() + 2500,
    });
    log("upload", id, `File ${file.name}. Demo mode: scores are synthetic.${clinicianUrgent ? " Clinician asked for urgency." : ""}`, actor);
    return build(studies[studies.length - 1], Date.now());
  },
  imageUrl() { return null; },
  heatmapUrl() { return null; },

  async review(id, body) {
    await sleep(150);
    const s = find(id);
    if (body.action === "mark_reviewed") {
      s.reviewed_ms = Date.now();
      log("mark_reviewed", id, "Reviewer acknowledged the AI output.", body.actor);
    } else if (body.action === "flag_disagreement") {
      s.flagged = true; s.reviewed_ms = Date.now();
      log("flag_disagreement", id, `${body.reason || "No reason"}. Human priority: ${body.human_priority || "n/a"}. ${body.notes || ""}`.trim(), body.actor);
    } else throw new Error("action must be mark_reviewed or flag_disagreement");
    return build(s, Date.now());
  },
  async urgentRequest(id, body) {
    await sleep(150);
    const s = find(id);
    s.clinician_urgent = true;
    log("urgent_request", id, body.note || "Clinician requested an urgent read.", body.actor);
    return build(s, Date.now());
  },
  async override(id, body) {
    await sleep(150);
    if (!body.reason || !body.reason.trim()) throw new Error("A reason is required for an override.");
    if (!PRIORITY_ORDER.hasOwnProperty(body.priority)) throw new Error("priority must be raised, normal or unsure");
    const s = find(id);
    s.override_priority = body.priority;
    log("override", id, `Priority set to ${body.priority}. Reason: ${body.reason}`, body.actor);
    return build(s, Date.now());
  },

  async getQueueSettings() { return { ...settings }; },
  async putQueueSettings(body) {
    await sleep(120);
    if (!["model", "fcfs"].includes(body.mode)) throw new Error("mode must be model or fcfs");
    const mw = Number(body.max_wait_minutes);
    if (!(mw >= 10 && mw <= 1440)) throw new Error("max_wait_minutes must be between 10 and 1440");
    settings = { mode: body.mode, max_wait_minutes: mw };
    log("queue_settings", null, `Mode ${body.mode}, max wait ${mw} min.`, body.actor);
    return { ...settings };
  },

  async getThreshold() { return { ...threshold }; },
  async previewThreshold(value) {
    await sleep(60);
    let P = 0, N = 0, tp = 0, fp = 0, unsure = 0;
    for (const { pos, score } of SAVED) {
      pos ? P++ : N++;
      if (score >= value) pos ? tp++ : fp++;
      else if (score >= value - BAND) unsure++;
    }
    return { value, abstention_rate: unsure / SAVED.length, sensitivity: tp / P, false_alarm_rate: fp / N, n: SAVED.length, n_positive: P };
  },
  async putThreshold(body) {
    await sleep(150);
    if (body.confirm !== true) throw new Error("confirm must be true to change the threshold.");
    const p = await this.previewThreshold(body.value);
    if (p.sensitivity < threshold.min_sensitivity_target) {
      throw new Error(`Sensitivity ${pct(p.sensitivity)}% is below the fixed minimum target of ${pct(threshold.min_sensitivity_target)}%.`);
    }
    threshold = { ...threshold, value: body.value };
    log("threshold_change", null, `Threshold set to ${body.value.toFixed(2)}.`, body.actor);
    return { ...threshold };
  },

  async runQueueRace(p) {
    await sleep(500);
    const arrivals = Number(p.arrivals_per_hour), reads = Number(p.reads_per_hour);
    const share = Number(p.urgent_share), hours = Number(p.hours);
    if (!(arrivals > 0 && reads > 0 && share >= 0 && share <= 1 && hours > 0)) throw new Error("Check the inputs.");
    if (arrivals * hours > 3000) throw new Error("Too many scans for the demo simulation (max 3000).");
    const r = rng(Number(p.seed) || 1);
    const gap = 60 / arrivals;
    const cases = [];
    let t = 0;
    for (;;) {
      t += -Math.log(1 - r()) * gap;
      if (t > hours * 60) break;
      const urgent = r() < share;
      const score = clamp01((urgent ? 0.72 : 0.22) + randn(r) * (urgent ? 0.18 : 0.16));
      const flag = score >= threshold.value ? "raised" : score >= threshold.value - BAND ? "unsure" : "normal";
      cases.push({ id: `SIM-${String(cases.length + 1).padStart(3, "0")}`, arrive: t, urgent, flag });
    }
    const service = 60 / reads;
    const run = (pick) => {
      const queue = [], done = [];
      let clock = 0, i = 0;
      while (done.length < cases.length) {
        while (i < cases.length && cases[i].arrive <= clock) queue.push(cases[i++]);
        if (!queue.length) { clock = cases[i].arrive; continue; }
        const c = queue.splice(pick(queue, clock), 1)[0];
        done.push({ ...c, start: clock, wait: clock - c.arrive });
        clock += service;
      }
      return done;
    };
    const fcfs = run(() => 0);
    const ct = run((q, clock) => {
      let best = 0, bestKey = null;
      q.forEach((c, k) => {
        const key = [clock - c.arrive >= settings.max_wait_minutes ? 0 : 1, PRIORITY_ORDER[c.flag], c.arrive];
        if (!bestKey || key[0] < bestKey[0] || (key[0] === bestKey[0] && (key[1] < bestKey[1] || (key[1] === bestKey[1] && key[2] < bestKey[2])))) { best = k; bestKey = key; }
      });
      return best;
    });
    const mean = (a) => (a.length ? a.reduce((x, y) => x + y.wait, 0) / a.length : 0);
    const r1 = (x) => Math.round(x * 10) / 10;
    const urg = (d) => d.filter((c) => c.urgent), oth = (d) => d.filter((c) => !c.urgent);
    const order = (d) => [...d].sort((a, b) => a.start - b.start).slice(0, 12).map((c) => ({ id: c.id, urgent: c.urgent, flag: c.flag, wait_min: r1(c.wait) }));
    return {
      assumptions: {
        arrivals_per_hour: arrivals, reads_per_hour: reads, urgent_share: share, hours, seed: Number(p.seed) || 1,
        readers: 1, threshold: threshold.value, max_wait_minutes: settings.max_wait_minutes,
        note: "Synthetic arrivals and scores in demo mode. A simulation is not clinical proof.",
      },
      n_cases: cases.length, n_urgent: urg(cases).length,
      fcfs: { urgent_wait_min: r1(mean(urg(fcfs))), other_extra_wait_min: 0, missed_urgent: null, wrong_flags: null },
      voxelsort: {
        urgent_wait_min: r1(mean(urg(ct))),
        other_extra_wait_min: r1(mean(oth(ct)) - mean(oth(fcfs))),
        missed_urgent: cases.filter((c) => c.urgent && c.flag === "normal").length,
        wrong_flags: cases.filter((c) => !c.urgent && c.flag === "raised").length,
      },
      read_order: { fcfs: order(fcfs), voxelsort: order(ct) },
    };
  },

  async sampleSize({ target_sensitivity, expected_sensitivity, confidence = 0.95, prevalence = 0.2 }) {
    const target = Number(target_sensitivity), exp = Number(expected_sensitivity), z = zFor(Number(confidence));
    if (!(target > 0 && target < 1 && exp > 0 && exp < 1)) throw new Error("Sensitivities must be between 0 and 1.");
    if (exp <= target) return { positives_needed: null, scans_needed: null, note: "Expected sensitivity must be higher than the target, otherwise no sample size can confirm it." };
    for (let n = 1; n <= 20000; n++) {
      if (wilson(Math.round(exp * n), n, z)[0] >= target) {
        return { positives_needed: n, scans_needed: Math.ceil(n / Number(prevalence || 0.2)), method: "wilson", note: "Assumes the true sensitivity equals the expected value." };
      }
    }
    return { positives_needed: null, scans_needed: null, note: "More than 20,000 positive scans would be needed." };
  },

  jobs: {},
  async startSiteReadiness(files, labelsFile, actor) {
    await sleep(300);
    const id = `job-${Date.now()}`;
    const labels = labelsFile ? await labelsFile.text() : "";
    this.jobs[id] = { created: Date.now(), n: files.length, size: files.reduce((a, f) => a + f.size, 0), labels };
    log("site_readiness_run", null, `${files.length} scans${labelsFile ? ", labels provided" : ", no labels"}.`, actor);
    return { job_id: id };
  },
  async getSiteReadiness(id) {
    await sleep(80);
    const j = this.jobs[id];
    if (!j) throw new Error("Job not found");
    if (Date.now() - j.created < 2000) return { status: "processing" };
    const target = threshold.min_sensitivity_target;
    const score = 0.12 + ((j.size % 997) / 997) * 0.5;
    const drift = { score: Math.round(score * 100) / 100, warning: score > 0.35, note: "Warning signal only, not a proven predictor of performance." };
    const rows = j.labels.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    const header = rows[0]?.toLowerCase().split(",") || [];
    const col = header.indexOf("label");
    const body = col >= 0 ? rows.slice(1) : rows;
    const positives = body.filter((l) => l.split(",")[col >= 0 ? col : l.split(",").length - 1].trim() === "1");
    if (!rows.length || !positives.length) {
      return { status: "done", drift, sensitivity: null, target, verdict: "not_enough_evidence", note: "No positive labels found. Add a CSV with a label column (1 = urgent finding)." };
    }
    const r = rng(j.size + positives.length);
    const hit = positives.filter(() => r() < 0.9).length;
    const [lo, hi] = wilson(hit, positives.length, 1.96);
    const sensitivity = { point: hit / positives.length, ci_low: lo, ci_high: hi, width: hi - lo, method: "wilson", n_positive: positives.length };
    let verdict;
    if (hi < target) verdict = "not_ready";
    else if (positives.length < 30 || lo < target) verdict = "not_enough_evidence";
    else verdict = drift.warning ? "use_with_caution" : "ready";
    return { status: "done", drift, sensitivity, target, verdict, note: "Demo mode: detections are simulated, not produced by a model." };
  },

  async oodCheck() { throw new Error("The out-of-distribution check needs the live backend and its real detector. It is not simulated in demo mode."); },
  async getEvidence() { return null; },

  async getAudit({ action, q, limit = 200 } = {}) {
    await sleep(80);
    return audit
      .filter((a) => (!action || a.action === action) && (!q || (a.study_id || "").toLowerCase().includes(q.toLowerCase())))
      .slice(0, limit);
  },
  async downloadAuditCsv() {
    const esc = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
    const head = ["id", "timestamp", "actor", "action", "study_id", "detail"];
    const csv = [head.join(","), ...audit.map((a) => head.map((h) => esc(a[h])).join(","))].join("\n");
    const url = URL.createObjectURL(new Blob([csv], { type: "text/csv" }));
    const a = document.createElement("a");
    a.href = url; a.download = "voxelsort-audit-demo.csv"; a.click();
    URL.revokeObjectURL(url);
  },
};
