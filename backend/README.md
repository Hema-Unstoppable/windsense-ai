# WindSense AI — Backend

The Python backend for **WindSense AI**, an OEM-agnostic Reliability Intelligence
Platform for wind turbines. It ingests SCADA data, runs ML fault prediction,
estimates Remaining Useful Life (RUL), ranks fleet risk using an FMECA engine,
and serves everything to the 4 dashboard screens via a REST API.

> **Status: verified working end-to-end** on the Wind Farm A (CARE to Compare)
> dataset — 5 turbines, 22 events, XGBoost classifier (ROC-AUC ≈ 0.94),
> live predictions, risk rankings, and Asset Health Certificates.

---

## 1. What it does (plain English)

```
  SCADA CSV  ─►  Normalisation  ─►  PostgreSQL  ─►  ML model  ─►  Risk engine  ─►  REST API  ─►  Dashboard
  (any OEM)      (canonical tags)   (time-series)   (XGBoost)     (FMECA)          (FastAPI)      (4 screens)
```

1. **Reads** your turbine SCADA data (CSV today, live SCADA/historian later).
2. **Normalises** every OEM's tag names to one canonical schema (OEM-agnostic).
3. **Stores** it in a time-series database (PostgreSQL in production, SQLite for instant local dev).
4. **Predicts** faults, failure mode (gearbox / generator / hydraulic / transformer), and RUL.
5. **Ranks** the whole fleet by risk = `fault_probability × FMECA_consequence × (1 − maintenance_coverage)`.
6. **Explains** every prediction with SHAP (human-readable signal drivers).
7. **Serves** it all as clean JSON for the 4 frontend screens + interactive docs at `/docs`.

---

## 2. Run it in 4 commands

> Requires Python 3.11+ (tested on 3.13). **Run everything from the `backend/` folder.**

```bash
cd backend

# 1. install dependencies (one time)
pip install -r requirements.txt

# 2. configure (one time) — copy the example, it already points at your CSV
copy .env.example .env        # Windows   (use: cp .env.example .env on Mac/Linux)

# 3. seed everything end-to-end: tables + data + train + predict + certificates
python -m scripts.seed_first_user

# 4. start the API
uvicorn app.main:app --reload
```

Then open **http://localhost:8000/docs** — an interactive API explorer where you
can click any endpoint and see live JSON.

**Faster re-runs** (reuse the trained model): `python -m scripts.seed_first_user --skip-train`
**Data only** (no ML): `python -m scripts.seed_first_user --skip-ml`

---

## 3. Folder structure

```
backend/
├── config/             # settings (.env) + database connection
│   ├── settings.py         pydantic-settings, all env vars typed
│   └── database.py         SQLAlchemy engine (PostgreSQL or SQLite)
├── models/             # database layer
│   ├── orm.py              9 ORM tables (portable PG/SQLite)
│   └── schemas.py          Pydantic JSON contracts (grouped by screen)
├── ml/                 # machine learning
│   ├── canonical.py        NORMALISATION ENGINE — OEM tag → canonical schema
│   ├── feature_engineering.py  targets, domain features, feature selection
│   ├── fmeca.py            FMECA RISK ENGINE — weights, risk score, RPN, RUL
│   ├── train_model.py      streaming balanced training + model versioning
│   ├── inference.py        scoring + SHAP + writes predictions
│   ├── evaluation.py       metrics + drift (PSI) monitoring
│   └── artifacts/          saved model bundles (.joblib)
├── services/           # business logic
│   ├── ingestion.py        CSV → database seeding
│   ├── risk_ranking.py     fleet risk-ranking recompute
│   ├── kpis.py             fleet KPI aggregation
│   └── asset_health.py     Asset Health Certificate builder
├── app/                # REST API
│   ├── main.py             FastAPI app + CORS
│   ├── deps.py             auth/tenant dependency (dev | Supabase JWT)
│   └── routers/            one router per screen
├── scripts/            # init_db.py, seed_first_user.py
├── schema.sql          # production PostgreSQL DDL (partitioning + indexes)
├── requirements.txt
└── .env.example
```

---

## 4. The 4 screens → API endpoints

| Screen | Endpoint | Returns |
|--------|----------|---------|
| **1 · Fleet Overview** | `GET /api/dashboard/fleet_overview` | KPIs, top-N risk-ranked turbines, anomaly alerts, data-source badge |
| **2 · Turbine Health** | `GET /api/turbines/{id}/health_summary` | health score, failure-mode predictions, component health, **SHAP drivers**, RUL, narrative |
| **3 · Anomaly Explorer** | `GET /api/turbines/{id}/timeseries?from=&to=&signals=` | 10-min SCADA series + anomaly markers (actual + predicted) |
| **3 · Risk Queue** | `GET /api/fleet/risk_ranking` | full FMECA-ranked maintenance queue with RPN + € exposure |
| **4 · Asset Health Certificate** | `GET /api/reports/asset_health_certificate?turbine_id=` | insurance-ready cert: component scores, RUL, sustainability KPIs, financial impact, compliance |

**ML operations**

| Endpoint | Purpose |
|----------|---------|
| `POST /api/ml/run_inference` | score one turbine or the whole fleet, write predictions + risk |
| `GET /api/ml/model` | active model version + metrics (from the model registry) |
| `POST /api/ml/retrain` | trigger background retraining (hot-reloads when done) |

Every JSON shape is defined in `models/schemas.py` so a frontend engineer can bind directly.
The data-source badge (`SCADA_LIVE` vs `CSV_UPLOAD` …) is returned on every screen response
so the UI can show the right icon.

---

## 5. Switching to PostgreSQL / Supabase (production)

SQLite is the zero-install default for local dev. For production, change **one line** in `.env`:

```ini
DATABASE_URL=postgresql+psycopg2://postgres:YOUR_PASSWORD@db.xxxx.supabase.co:5432/postgres
```

Then run the optimized schema (time-series partitioning, indexes, optional TimescaleDB):

```bash
psql "$DATABASE_URL" -f schema.sql
python -m scripts.seed_first_user
```

For very large fleets, `schema.sql` includes the TimescaleDB hypertable upgrade path
(compression + retention policies) at the bottom.

---

## 6. Multi-tenant & onboarding a new client/OEM

* **Multi-tenant** — every table carries `user_id`. Every query is scoped by it.
  In `AUTH_MODE=jwt`, the tenant is resolved from the Supabase JWT (`deps.py`); the
  same login system your frontend already uses.
* **New OEM feed** — add one mapping dict to `ml/canonical.py` (`PROFILES`). The
  database, ML, risk engine, and API are untouched. That is the OEM-agnostic promise
  implemented as code.

---

## 7. Production / scalability roadmap

| Concern | Built today | Next step |
|---------|-------------|-----------|
| **Background training** | `POST /api/ml/retrain` uses FastAPI BackgroundTasks | Move to **Celery / RQ + Redis** for durable job queues |
| **Batch inference** | `run_inference` scores the fleet on demand | Schedule every 10 min via **Celery beat / APScheduler** as live SCADA arrives |
| **Model versioning** | `model_registry` table + versioned `.joblib` bundles + `latest.joblib` pointer | Add **MLflow** for experiment tracking + artifact store (S3) |
| **Monitoring** | `evaluation.population_stability_index()` for drift; metrics persisted per version | Wire **Prometheus/Grafana** + automated drift-triggered retraining |
| **Logging** | Uvicorn structured logs | Centralise (Datadog / Loki); add request tracing |
| **Auth** | dev mode + Supabase JWT verification | Row-Level Security in Postgres for defense-in-depth |
| **Scale** | partitioned `scada_timeseries` by month | TimescaleDB hypertables + read replicas |

---

## 8. Retraining / tuning

```bash
# bigger, more accurate model (more rows per fault class)
python -m ml.train_model --rows-per-class 3000

# try Random Forest instead of XGBoost
python -m ml.train_model --algorithm random_forest
```

`MAX_SCADA_ROWS` in `.env` caps how much SCADA is loaded locally (keeps SQLite fast).
Set it to `0` to load everything on PostgreSQL.
```
