import React from "react";

// Illustrative placeholder drawing used ONLY in demo mode. It is not a patient image.
export default function DemoXray({ heatmap, opacity, zoom }) {
  return <div className="xray-stage"><div className="xray-label">DEMO IMAGE · NOT A PATIENT SCAN</div><svg className="xray-art" viewBox="0 0 500 570" style={{transform:`scale(${zoom})`}} role="img" aria-label="Illustrative chest X-ray placeholder, not a real medical image">
    <defs><radialGradient id="film" cx="50%" cy="45%" r="65%"><stop offset="0" stopColor="#4c5666"/><stop offset="1" stopColor="#121b2b"/></radialGradient><linearGradient id="bone" x1="0" x2="1"><stop stopColor="#9ba9b8" stopOpacity=".55"/><stop offset=".5" stopColor="#e2e9f0" stopOpacity=".9"/><stop offset="1" stopColor="#8a9aab" stopOpacity=".5"/></linearGradient><radialGradient id="lung" cx="50%" cy="30%" r="70%"><stop stopColor="#172638"/><stop offset="1" stopColor="#26374a"/></radialGradient><filter id="blur"><feGaussianBlur stdDeviation="15"/></filter></defs>
    <rect x="0" y="0" width="500" height="570" rx="12" fill="url(#film)"/>
    <g opacity=".5" stroke="#a9b6c5" strokeWidth="2" fill="none"><path d="M250 52 L250 468"/><path d="M190 95 Q125 120 112 198 Q95 270 122 365"/><path d="M310 95 Q375 120 388 198 Q405 270 378 365"/></g>
    <path d="M225 35 Q250 25 275 35 L290 95 Q250 117 210 95Z" fill="url(#bone)" opacity=".7"/>
    <path d="M232 80 C197 60 145 89 125 139 C100 204 116 302 145 366 C164 408 200 418 221 382 C239 344 237 265 239 196Z" fill="url(#lung)" stroke="#9eacbb" strokeOpacity=".35" strokeWidth="2"/>
    <path d="M268 80 C303 60 355 89 375 139 C400 204 384 302 355 366 C336 408 300 418 279 382 C261 344 263 265 261 196Z" fill="url(#lung)" stroke="#9eacbb" strokeOpacity=".35" strokeWidth="2"/>
    <path d="M249 95 C237 160 237 228 244 295 C246 325 226 345 221 376 C218 402 238 429 257 432 C279 430 296 404 286 376 C274 337 262 306 265 265 L267 95Z" fill="#b8c4d0" opacity=".72"/>
    <path d="M250 115 Q210 115 174 143 M250 115 Q290 115 326 143" stroke="#d0d8e1" strokeOpacity=".6" strokeWidth="8" fill="none" strokeLinecap="round"/>
    {Array.from({length:8},(_,i)=><g key={i} fill="none" stroke="url(#bone)" strokeWidth="7" opacity=".6"><path d={`M${220-i*2} ${133+i*29} Q${166-i*5} ${120+i*31} ${133+i*2} ${151+i*27}`}/><path d={`M${280+i*2} ${133+i*29} Q${334+i*5} ${120+i*31} ${367-i*2} ${151+i*27}`}/></g>)}
    <path d="M157 390 Q250 420 343 390" stroke="#bdc8d4" strokeWidth="7" opacity=".5" fill="none"/>
    <path d="M175 438 Q250 460 325 438" stroke="#9daab9" strokeWidth="3" opacity=".45" fill="none"/>
    {heatmap && <g opacity={opacity/100}><ellipse cx="315" cy="274" rx="72" ry="85" fill="#f24b5b" filter="url(#blur)"/><ellipse cx="315" cy="274" rx="49" ry="61" fill="#ffad45" opacity=".8"/><ellipse cx="315" cy="274" rx="28" ry="35" fill="#ffe58a" opacity=".8"/></g>}
    <text x="20" y="30" fill="#dce4ed" fontSize="12" letterSpacing="2">CHEST · AP</text><text x="20" y="548" fill="#a7b5c5" fontSize="10" letterSpacing="1.5">ILLUSTRATIVE PLACEHOLDER</text>
  </svg><div className="xray-scale"><span>R</span><span>0</span><span>L</span></div></div>;
}
