import React from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  ArrowDownUp, ChevronRight, FileSearch, Gauge, History, Layers, Menu, ShieldCheck, Upload, X, Radar,
} from "lucide-react";
import { isLive } from "../services/api.js";
import { getActor, setActor } from "../lib/format.js";

export const navItems = [
  { to: "/", label: "Worklist", icon: Layers, end: true },
  { to: "/upload", label: "Upload study", icon: Upload },
  { to: "/safety-speed", label: "Safety vs speed", icon: Gauge },
  { to: "/queue-race", label: "Queue Race", icon: ArrowDownUp },
  { to: "/site-readiness", label: "Site readiness", icon: ShieldCheck },
  { to: "/evidence", label: "Evidence", icon: FileSearch },
  { to: "/audit", label: "Audit log", icon: History },
];

export default function Shell({ children }) {
  const location = useLocation();
  const [mobileNav, setMobileNav] = React.useState(false);
  const [actor, setActorState] = React.useState(getActor());
  const current = navItems.find((n) => (n.end ? location.pathname === n.to : location.pathname.startsWith(n.to)));
  const title = location.pathname.startsWith("/study/") ? "Study review" : current?.label || "Worklist";

  return (
    <div className="app-shell">
      <aside className={`sidebar ${mobileNav ? "mobile-open" : ""}`}>
        <div className="brand">
          <div className="brand-mark"><Radar size={21} /></div>
          <div><strong>VoxelSort</strong><span>READING QUEUE PRIORITISER</span></div>
          <button className="icon-btn sidebar-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={17} /></button>
        </div>
        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav">
          {navItems.map(({ to, label, icon: Icon, end }) => (
            <NavLink key={to} to={to} end={end} onClick={() => setMobileNav(false)} className={({ isActive }) => `nav-link ${isActive ? "active" : ""}`}>
              <Icon size={18} /><span>{label}</span>
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="prototype-card">
            <div className="prototype-icon"><ShieldCheck size={16} /></div>
            <div><strong>Decision support only</strong><span>A clinician reads every scan</span></div>
          </div>
          <label className="actor-box">
            <span>Your name (optional)</span>
            <input value={actor} onChange={(e) => { setActorState(e.target.value); setActor(e.target.value); }} placeholder="Shown in the audit log" maxLength={40} />
          </label>
        </div>
      </aside>
      {mobileNav && <button className="mobile-scrim" onClick={() => setMobileNav(false)} aria-label="Close menu" />}
      <div className="app-main">
        <header className="topbar">
          <div className="topbar-left">
            <button className="icon-btn mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={20} /></button>
            <div className="breadcrumbs"><span>VoxelSort</span><ChevronRight size={14} /><strong>{title}</strong></div>
          </div>
          <div className="topbar-right">
            <div className={`environment ${isLive ? "live" : "demo"}`}>
              <span className="environment-dot" /> {isLive ? "LIVE BACKEND" : "DEMO DATA · NO BACKEND"}
            </div>
          </div>
        </header>
        {!isLive && (
          <div className="demo-strip">
            Demo mode: studies, scores and simulations are synthetic and run in your browser. Set <code>VITE_API_URL</code> to connect the real backend.
          </div>
        )}
        <main className="page-content">{children}</main>
        <footer className="app-footer">
          <span>VoxelSort <span className="footer-sep">·</span> Research prototype</span>
          <span><ShieldCheck size={14} /> AI decision support only · Human review required</span>
        </footer>
      </div>
    </div>
  );
}
