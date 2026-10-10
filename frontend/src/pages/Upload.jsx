import React from "react";
import { useNavigate } from "react-router-dom";
import { AlertCircle, CheckCircle2, FileImage, Upload as UploadIcon, ShieldCheck, X } from "lucide-react";
import { api, isLive } from "../services/api.js";
import { getActor } from "../lib/format.js";
import { Badge, Notice, SectionHeading } from "../components/ui.jsx";

export default function Upload() {
  const navigate = useNavigate();
  const [file, setFile] = React.useState(null);
  const [urgent, setUrgent] = React.useState(false);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState("");
  const inputRef = React.useRef(null);

  const submit = async () => {
    if (!file) return setError("Choose a file first.");
    setBusy(true); setError("");
    try {
      const study = await api.uploadStudy(file, urgent, getActor() || undefined);
      navigate(`/study/${study.study_id}`);
    } catch (e) { setError(e.message); setBusy(false); }
  };

  return (
    <>
      <SectionHeading eyebrow="STUDY INTAKE" title="Upload study" description="Add a de-identified chest X-ray. It joins the queue straight away and the model scores it in the background." action={<Badge tone={isLive ? "success" : "warning"}>{isLive ? "Live backend" : "Demo: synthetic scores"}</Badge>} />
      <div className="upload-layout">
        <section className="panel upload-panel">
          <div className="upload-zone" onDragOver={(e) => e.preventDefault()} onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files?.[0] || null); setError(""); }}>
            <div className="upload-icon"><FileImage size={26} /></div>
            <h2>Drop a study image here</h2>
            <p>or browse files from your device</p>
            <button className="btn btn-secondary" onClick={() => inputRef.current?.click()}><UploadIcon size={16} /> Browse files</button>
            <input ref={inputRef} type="file" accept=".png,.jpg,.jpeg,.dcm,.dicom" hidden onChange={(e) => { setFile(e.target.files?.[0] || null); setError(""); }} />
            <div className="file-types">PNG, JPG or DICOM · De-identified data only</div>
          </div>
          {file && (
            <div className="selected-file">
              <FileImage size={19} />
              <div><strong>{file.name}</strong><span>{(file.size / 1024).toFixed(0)} KB</span></div>
              <button className="icon-btn" onClick={() => setFile(null)} aria-label="Remove file"><X size={17} /></button>
            </div>
          )}
          <label className="check-row">
            <input type="checkbox" checked={urgent} onChange={(e) => setUrgent(e.target.checked)} />
            <span><strong>A clinician has asked for this to be read urgently</strong><small>A clinician's request always goes to the top of the queue, whatever the model says.</small></span>
          </label>
          {error && <Notice>{error}</Notice>}
          <button className="btn btn-primary btn-block upload-submit" disabled={busy} onClick={submit}><UploadIcon size={16} /> {busy ? "Adding..." : "Add to queue"}</button>
        </section>
        <aside className="panel upload-info">
          <div className="info-heading"><ShieldCheck size={20} /><h2>Before uploading</h2></div>
          <ul className="check-list">
            <li><CheckCircle2 size={16} /> Use public, de-identified images only.</li>
            <li><CheckCircle2 size={16} /> Do not upload identifiable patient information.</li>
            <li><CheckCircle2 size={16} /> The scan is queued first and scored in the background.</li>
            <li><CheckCircle2 size={16} /> A scan the model cannot place is marked "AI unsure", never "routine".</li>
          </ul>
          <div className="notice-box"><AlertCircle size={17} /><p><strong>{isLive ? "Sent to the backend" : "Demo mode"}</strong><br />{isLive ? "The image is sent to the VoxelSort backend for scoring." : "Nothing leaves your browser. Scores are synthetic."}</p></div>
        </aside>
      </div>
    </>
  );
}
