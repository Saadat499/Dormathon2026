import React from "react";
import { AlertTriangle, CheckCircle2, Info, SlidersHorizontal } from "lucide-react";
import { api } from "../services/api.js";
import { getActor, pct } from "../lib/format.js";
import { Badge, Banner, Loading, Modal, Notice, SectionHeading } from "../components/ui.jsx";

export default function SafetySpeed() {
  const [current, setCurrent] = React.useState(null);
  const [value, setValue] = React.useState(0.42);
  const [preview, setPreview] = React.useState(null);
  const [error, setError] = React.useState("");
  const [msg, setMsg] = React.useState(null);
  const [confirm, setConfirm] = React.useState(false);
  const [ack, setAck] = React.useState(false);
  const [busy, setBusy] = React.useState(false);

  React.useEffect(() => {
    api.getThreshold().then((t) => { setCurrent(t); setValue(t.value); }).catch((e) => setError(e.message));
  }, []);

  // Recalculate from saved predictions whenever the slider settles.
  React.useEffect(() => {
    if (!current) return undefined;
    const t = setTimeout(() => {
      api.previewThreshold(value).then((p) => { setPreview(p); setError(""); }).catch((e) => setError(e.message));
    }, 150);
    return () => clearTimeout(t);
  }, [value, current]);

  if (!current) return <Loading error={error} what="threshold" />;
  const target = current.min_sensitivity_target;
  const belowTarget = preview && preview.sensitivity < target;
  const changed = Math.abs(value - current.value) > 0.0001;

  const apply = async () => {
    setBusy(true);
    try {
      const t = await api.putThreshold({ value, confirm: true, actor: getActor() || undefined });
      setCurrent(t); setConfirm(false); setAck(false);
      setMsg({ tone: "success", text: `Threshold changed to ${t.value.toFixed(2)} and logged.` });
    } catch (e) { setMsg({ tone: "warning", text: e.message }); setConfirm(false); }
    setBusy(false);
  };

  return (
    <>
      <SectionHeading
        eyebrow="OPERATING POINT"
        title="Safety vs speed"
        description="Move the threshold and see what happens to missed urgent scans, false alarms and 'AI unsure' scans. Numbers are recalculated from saved predictions."
        action={<Badge tone="neutral">Current threshold {current.value.toFixed(2)}</Badge>}
      />
      {msg && <Notice tone={msg.tone}>{msg.text}</Notice>}
      {error && <Notice>{error}</Notice>}
      <div className="safety-layout">
        <section className="panel threshold-panel">
          <div className="panel-header"><div><h2>Threshold</h2><p>Lower flags more scans as urgent. Higher flags fewer.</p></div><SlidersHorizontal size={19} className="muted-icon" /></div>
          <div className="threshold-value"><span>Decision threshold</span><strong>{value.toFixed(2)}</strong></div>
          <input className="threshold-slider" type="range" min="0.1" max="0.9" step="0.01" value={value} onChange={(e) => { setValue(Number(e.target.value)); setMsg(null); }} />
          <div className="range-labels"><span>More alarms, fewer misses</span><span>Fewer alarms, more misses</span></div>
          <div className="threshold-hint"><Info size={16} /><p>The minimum sensitivity target ({pct(target)}) is fixed first. False alarms are then minimised at that target. You cannot apply a threshold below it.</p></div>
          <button className="btn btn-primary btn-block" disabled={!changed || belowTarget || !preview} onClick={() => setConfirm(true)}>Apply this threshold...</button>
          {belowTarget && <Notice>At {value.toFixed(2)} sensitivity falls below the {pct(target)} target, so this threshold cannot be applied.</Notice>}
        </section>

        <section className="metrics-area">
          <div className="metric-card">
            <span>Sensitivity (urgent scans caught)</span>
            <strong>{preview ? pct(preview.sensitivity, 1) : "..."}</strong>
            <div className="metric-track"><i style={{ width: `${(preview?.sensitivity || 0) * 100}%` }} /><b className="target-mark" style={{ left: `${target * 100}%` }} title="Minimum target" /></div>
            <small>Target {pct(target)}</small>
          </div>
          <div className="metric-card">
            <span>False alarms (routine flagged urgent)</span>
            <strong>{preview ? pct(preview.false_alarm_rate, 1) : "..."}</strong>
            <div className="metric-track amber-track"><i style={{ width: `${(preview?.false_alarm_rate || 0) * 100}%` }} /></div>
            <small>Share of routine scans</small>
          </div>
          <div className="metric-card">
            <span>"AI unsure" rate</span>
            <strong>{preview ? pct(preview.abstention_rate, 1) : "..."}</strong>
            <div className="metric-track teal-track"><i style={{ width: `${(preview?.abstention_rate || 0) * 100}%` }} /></div>
            <small>Moved up, never cleared</small>
          </div>
          <div className="panel comparison-panel">
            <h2>What this means in practice</h2>
            {preview && (
              <ul className="limitation-list">
                <li>About <strong>{Math.round((1 - preview.sensitivity) * 1000)}</strong> of every 1,000 urgent scans are not flagged as urgent review at this setting.</li>
                <li>About <strong>{Math.round(preview.false_alarm_rate * 1000)}</strong> of every 1,000 routine scans are flagged as urgent by mistake.</li>
                <li>Error never reaches zero. Lowering the threshold trades missed scans for false alarms.</li>
                <li>Computed from {preview.n} saved predictions. Scans that are not flagged stay in the queue; nothing is cleared.</li>
              </ul>
            )}
          </div>
        </section>
      </div>
      <Banner icon={AlertTriangle} title="Thresholds are a clinical decision">
        Changing the threshold changes who is read first. It needs a confirmation and is recorded in the audit log. With no login, the confirmation and the log are the only controls.
      </Banner>

      {confirm && (
        <Modal onClose={() => setConfirm(false)}>
          <div className="modal-icon amber"><AlertTriangle size={20} /></div>
          <h2>Change the threshold?</h2>
          <p>From <strong>{current.value.toFixed(2)}</strong> to <strong>{value.toFixed(2)}</strong>. Expected sensitivity {pct(preview.sensitivity, 1)}, false alarms {pct(preview.false_alarm_rate, 1)}.</p>
          <label className="check-row"><input type="checkbox" checked={ack} onChange={(e) => setAck(e.target.checked)} /><span><strong>A clinical lead has approved this change</strong><small>It takes effect for all scans in the queue straight away.</small></span></label>
          <div className="modal-actions"><button className="btn btn-secondary" onClick={() => setConfirm(false)}>Cancel</button><button className="btn btn-primary" disabled={!ack || busy} onClick={apply}><CheckCircle2 size={16} /> Confirm change</button></div>
        </Modal>
      )}
    </>
  );
}
