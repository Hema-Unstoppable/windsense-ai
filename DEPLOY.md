# WindSense AI — Deployment Guide

Go live in ~45 minutes. Three pieces:

```
   Vercel (static frontend)  ─►  Render (FastAPI backend)  ─►  Supabase (Postgres + Auth)
   landing-page/                 backend/                      tegudhuanbwczeoykkxp
```

All three have free tiers. Auth = real Supabase JWT (ES256) — verified server-side.

---

## Step 0 — Accounts you need
- GitHub (to host the code Render & Vercel pull from)
- Render: https://render.com  (backend)
- Vercel: https://vercel.com  (frontend)
- Supabase: already have it ✅

---

## Step 1 — Get your Supabase Postgres connection string
Supabase Dashboard → **Project Settings → Database → Connection string → URI**.
Choose the **"Session pooler"** (or "Transaction pooler") URI. It looks like:
```
postgresql://postgres.tegudhuanbwczeoykkxp:[YOUR-PASSWORD]@aws-0-eu-west-1.pooler.supabase.com:5432/postgres
```
Keep it secret. You'll paste it in two places: the local seed (Step 2) and Render (Step 4).

> Tip: convert it to SQLAlchemy form by prefixing `postgresql+psycopg2://` (the app accepts plain `postgresql://` too).

---

## Step 2 — Seed your Supabase database (run ONCE, locally)
This loads the schema, Farm A data, trains the model, runs inference, and writes
everything to Supabase. (Render itself never needs the 824 MB CSV.)

```bash
cd backend
# point at Supabase for this one run:
#   Windows PowerShell:  $env:DATABASE_URL="postgresql+psycopg2://...":  then run
#   bash:                DATABASE_URL="postgresql+psycopg2://..." \
DATABASE_URL="<your-supabase-uri>" AUTH_MODE=jwt python -m scripts.seed_first_user
```
This also produces `backend/ml/artifacts/latest.joblib`, `calibrator.joblib`,
`validation_report.json` — which are committed and shipped to Render.

> If you prefer, run `psql "<uri>" -f schema.sql` first for the optimized
> partitioned schema; otherwise the app auto-creates portable tables.

---

## Step 3 — Push the code to GitHub
From the `Coding/` folder (this is the repo root — `.gitignore` already excludes
`Data/`, `*.db`, `.env`):
```bash
git init
git add .
git commit -m "WindSense AI — initial deploy"
git branch -M main
git remote add origin https://github.com/<you>/windsense-ai.git
git push -u origin main
```

---

## Step 4 — Deploy the backend on Render
1. Render → **New → Blueprint** → connect your repo. It reads `render.yaml`.
2. When prompted, set the secret env var:
   - `DATABASE_URL` = your Supabase URI (from Step 1)
3. Confirm the pre-filled vars: `AUTH_MODE=jwt`, `SUPABASE_URL=https://tegudhuanbwczeoykkxp.supabase.co`, `CORS_ORIGINS=*`.
4. Deploy. You'll get a URL like `https://windsense-api.onrender.com`.
5. Test: open `https://windsense-api.onrender.com/api/health` → `{"status":"healthy",...}`.

> Free instances sleep after 15 min idle (≈30 s cold start on first hit) — fine for demos.

---

## Step 5 — Point the frontend at the backend, then deploy on Vercel
1. Edit `landing-page/config.js` → set:
   ```js
   window.WS_API_ROOT = "https://windsense-api.onrender.com";   // your Render URL
   ```
   Commit & push.
2. Vercel → **New Project** → import the repo → set **Root Directory = `landing-page`** →
   Framework preset: **Other** (it's static) → Deploy.
3. You'll get a URL like `https://windsense-ai.vercel.app`.

---

## Step 6 — Connect the pieces (security)
1. **Render** → env `CORS_ORIGINS` → change `*` to your Vercel domain
   (`https://windsense-ai.vercel.app`) → save (redeploys).
2. **Supabase** → Authentication → **URL Configuration** → set **Site URL** to your
   Vercel domain and add it under **Redirect URLs**.
3. **Google OAuth** (if used) → Google Cloud Console → add the Vercel domain to
   Authorised JavaScript origins; the Supabase callback URI is already registered.

---

## Step 7 — Create the operator login
Supabase → Authentication → **Users → Add user**:
- Email: `operator_farma@windsense.ai`  (must match the seeded tenant)
- Password: your choice

---

## Step 8 — Test checklist
- [ ] `…onrender.com/api/health` returns healthy
- [ ] Vercel site loads the landing page
- [ ] Log in as `operator_farma@windsense.ai`
- [ ] Dashboard pill shows **"Live data · API connected"**, 5 turbines
- [ ] ML Predictions, Risk Queue (EEL), SCADA charts, Certificates (PDF download), ML Validation, Onboarding all render
- [ ] A *different* login sees an empty tenant (isolation works)

---

## Known limitations on the hosted demo (by design)
- The 824 MB CSV is **not** on the server, so **"Re-run validation"** and
  **"Onboarding → Validate"** (which read the raw CSV) won't run in prod. The
  **pre-computed** validation report + seeded data + live ML inference all work.
- To refresh validation/onboarding in prod later, host the CSV in object storage
  (S3/Supabase Storage) and point `CSV_PATH` at it.
