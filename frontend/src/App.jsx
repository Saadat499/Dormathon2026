import React from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import Shell from "./components/Shell.jsx";
import Worklist from "./pages/Worklist.jsx";
import Upload from "./pages/Upload.jsx";
import StudyReview from "./pages/StudyReview.jsx";
import SafetySpeed from "./pages/SafetySpeed.jsx";
import QueueRace from "./pages/QueueRace.jsx";
import SiteReadiness from "./pages/SiteReadiness.jsx";
import Evidence from "./pages/Evidence.jsx";
import AuditLog from "./pages/AuditLog.jsx";

// No login and no dashboard: the worklist is the home screen.
export default function App() {
  return (
    <Shell>
      <Routes>
        <Route path="/" element={<Worklist />} />
        <Route path="/upload" element={<Upload />} />
        <Route path="/study/:studyId" element={<StudyReview />} />
        <Route path="/safety-speed" element={<SafetySpeed />} />
        <Route path="/queue-race" element={<QueueRace />} />
        <Route path="/site-readiness" element={<SiteReadiness />} />
        <Route path="/evidence" element={<Evidence />} />
        <Route path="/audit" element={<AuditLog />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </Shell>
  );
}
