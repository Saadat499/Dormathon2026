# VoxelSort frontend

Chest X-ray reading-queue prioritiser. Decision support only: a clinician reads every scan. There is no login and no dashboard, just the working screens: Worklist, Upload, Study review, Safety vs speed, Queue Race, Site readiness, Evidence and Audit log.

## Run

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # production build in dist/
```

## Demo vs live

- **Demo mode (default):** `VITE_API_URL` is empty. Everything runs in the browser on synthetic data (`src/services/mock.js`). A "DEMO DATA" pill and banner say so on every page.
- **Live mode:** copy `.env.example` to `.env` and set `VITE_API_URL` to the backend URL, then restart. The same pages call the API in `docs/API_CONTRACT.md`. A "LIVE BACKEND" pill appears.

The Evidence page shows "Not available yet" for anything the backend does not supply. It never invents numbers.

## Structure

- `src/services/api.js` picks the live client or the mock. Both expose the same methods.
- `src/pages/` one file per screen. `src/components/` shared UI. `src/lib/` formatting and polling hooks.
- Polling (not WebSockets) keeps the worklist and audit log fresh.

Everything shown in demo mode is synthetic and not for clinical use.
