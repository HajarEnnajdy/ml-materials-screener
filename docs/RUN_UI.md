# Running the Material Screener UI

The app has two parts that must BOTH be running at the same time:
the Python backend (API + models) and the React frontend (the web page).

## 1. Backend (API) — terminal 1

From the project root:

```bash
python -m uvicorn api.main:app --port 8001
```

Leave it running. It serves predictions + structure lookups on
http://127.0.0.1:8001. Needs `MP_API_KEY` in `.env` (already set) for the
3D structure fetches.

Check it's up:  open http://127.0.0.1:8001/api/health → should say `{"status":"ok"}`

## 2. Frontend (web page) — terminal 2

In a SECOND terminal:

```bash
cd frontend
npm install     # first time only
npm run dev
```

It prints a local URL (usually http://localhost:5173). Open that in your
browser.

## Using it
- **Dataset sample** — scores the first N real Materials Project compounds.
- **From elements** — generates candidate formulas from an element set.
- **Custom formulas** — scores formulas you type in.
- Click any result card to see its 3D crystal structure, chemical name,
  and space group (for compounds that exist in Materials Project).

## Ports / notes
- Frontend talks to the backend at port **8001** (set in
  `frontend/src/App.jsx` and `frontend/src/StructureModal.jsx` as
  `API_BASE`). If you ever change the backend port, update both files.
- To stop either server: press Ctrl+C in its terminal.
