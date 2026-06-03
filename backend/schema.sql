-- ════════════════════════════════════════════════════════════════════
--  WindSense AI — PostgreSQL Schema (production reference DDL)
--
--  This is the production-grade PostgreSQL schema with time-series
--  partitioning and indexing. The SQLAlchemy ORM (backend/models/orm.py)
--  creates a portable version automatically for local SQLite dev; this
--  file documents the optimized PostgreSQL deployment and is what you run
--  on Supabase / managed Postgres.
--
--  Run:  psql "$DATABASE_URL" -f backend/schema.sql
--
--  Optional (recommended for very large fleets): install TimescaleDB and
--  convert scada_timeseries to a hypertable (see note at bottom).
-- ════════════════════════════════════════════════════════════════════

-- ── Enums ────────────────────────────────────────────────────────────
DO $$ BEGIN
    CREATE TYPE data_source_enum AS ENUM ('SCADA_LIVE', 'CSV_UPLOAD', 'API_PULL', 'HISTORIAN_PI', 'SIMULATED');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE event_label_enum AS ENUM ('normal', 'anomaly', 'maintenance', 'curtailment');
EXCEPTION WHEN duplicate_object THEN null; END $$;

DO $$ BEGIN
    CREATE TYPE risk_class_enum AS ENUM ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL');
EXCEPTION WHEN duplicate_object THEN null; END $$;


-- ── 1. USERS / TENANTS ───────────────────────────────────────────────
-- Every other table carries user_id for logical multi-tenant isolation.
CREATE TABLE IF NOT EXISTS users (
    id              BIGSERIAL PRIMARY KEY,
    auth_uid        TEXT UNIQUE,                 -- Supabase auth.uid (nullable in dev)
    email           TEXT UNIQUE NOT NULL,
    display_name    TEXT,
    company         TEXT,
    plan            TEXT DEFAULT 'pilot',        -- pilot | standard | advanced | enterprise
    created_at      TIMESTAMPTZ DEFAULT now()
);

-- ── 2. SITES (wind farms) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS sites (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name            TEXT NOT NULL,
    country         TEXT,
    latitude        DOUBLE PRECISION,
    longitude       DOUBLE PRECISION,
    data_source     data_source_enum DEFAULT 'CSV_UPLOAD',
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_sites_user ON sites(user_id);

-- ── 3. TURBINES ──────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS turbines (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    site_id         BIGINT NOT NULL REFERENCES sites(id) ON DELETE CASCADE,
    external_ref    TEXT,                        -- original asset_id from SCADA (e.g. "13")
    name            TEXT NOT NULL,               -- e.g. "WT-013"
    oem             TEXT,                        -- Vestas / GE / Enercon ...
    model           TEXT,                        -- V90 / 2.5XL ...
    rated_power_kw  DOUBLE PRECISION,
    commissioned_on DATE,
    data_source     data_source_enum DEFAULT 'CSV_UPLOAD',
    created_at      TIMESTAMPTZ DEFAULT now(),
    UNIQUE (user_id, site_id, external_ref)
);
CREATE INDEX IF NOT EXISTS ix_turbines_user ON turbines(user_id);
CREATE INDEX IF NOT EXISTS ix_turbines_site ON turbines(site_id);

-- ── 4. SCADA TIME-SERIES (PARTITIONED BY TIME) ───────────────────────
-- Native PostgreSQL declarative range partitioning by month.
-- Canonical, OEM-normalised signals are stored in the JSONB `signals`
-- column so the same table serves any turbine make/model.
CREATE TABLE IF NOT EXISTS scada_timeseries (
    id              BIGSERIAL,
    user_id         BIGINT NOT NULL,
    turbine_id      BIGINT NOT NULL,
    ts              TIMESTAMPTZ NOT NULL,
    status_type_id  SMALLINT,                    -- 0..5 operating status
    data_source     data_source_enum DEFAULT 'CSV_UPLOAD',
    -- frequently-queried hot signals as real columns (fast filtering/plots):
    wind_speed      DOUBLE PRECISION,
    power_kw        DOUBLE PRECISION,
    rotor_rpm       DOUBLE PRECISION,
    gearbox_oil_temp DOUBLE PRECISION,
    gen_bearing_de_temp DOUBLE PRECISION,
    nacelle_temp    DOUBLE PRECISION,
    ambient_temp    DOUBLE PRECISION,
    -- everything else (full canonical signal set) as JSONB:
    signals         JSONB,
    PRIMARY KEY (id, ts)
) PARTITION BY RANGE (ts);

-- Composite index that powers the core access pattern: one turbine, a time window.
CREATE INDEX IF NOT EXISTS ix_scada_turbine_ts ON scada_timeseries (turbine_id, ts DESC);
CREATE INDEX IF NOT EXISTS ix_scada_user_ts    ON scada_timeseries (user_id, ts DESC);

-- Example monthly partitions (create as data ranges require; automate with pg_partman).
CREATE TABLE IF NOT EXISTS scada_timeseries_2022 PARTITION OF scada_timeseries
    FOR VALUES FROM ('2022-01-01') TO ('2023-01-01');
CREATE TABLE IF NOT EXISTS scada_timeseries_2023 PARTITION OF scada_timeseries
    FOR VALUES FROM ('2023-01-01') TO ('2024-01-01');
CREATE TABLE IF NOT EXISTS scada_timeseries_default PARTITION OF scada_timeseries DEFAULT;

-- ── 5. EVENTS (alarms / failures / maintenance) ──────────────────────
CREATE TABLE IF NOT EXISTS events (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    turbine_id      BIGINT NOT NULL REFERENCES turbines(id) ON DELETE CASCADE,
    external_event_id TEXT,                       -- original event_id
    label           event_label_enum NOT NULL DEFAULT 'normal',
    description     TEXT,                          -- failure mode text
    component       TEXT,                          -- gearbox / generator / hydraulic / transformer
    start_time      TIMESTAMPTZ,
    end_time        TIMESTAMPTZ,
    data_source     data_source_enum DEFAULT 'CSV_UPLOAD',
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_events_turbine ON events(turbine_id, start_time DESC);
CREATE INDEX IF NOT EXISTS ix_events_user    ON events(user_id);

-- ── 6. ML PREDICTIONS (per turbine, per timestamp/window) ────────────
CREATE TABLE IF NOT EXISTS ml_predictions (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    turbine_id      BIGINT NOT NULL REFERENCES turbines(id) ON DELETE CASCADE,
    ts              TIMESTAMPTZ NOT NULL,          -- window the prediction refers to
    model_version   TEXT NOT NULL,
    fault_probability DOUBLE PRECISION,            -- P(anomaly) 0..1
    predicted_class INTEGER,                       -- 0 normal,1 gen,2 gearbox,3 hydraulic,4 transformer
    predicted_component TEXT,
    class_probabilities JSONB,                     -- {"0":.1,"2":.7,...}
    rul_days        DOUBLE PRECISION,              -- remaining useful life estimate
    confidence      DOUBLE PRECISION,
    shap_top_features JSONB,                       -- [{"feature":..,"value":..,"contribution":..}]
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_pred_turbine_ts ON ml_predictions(turbine_id, ts DESC);
CREATE INDEX IF NOT EXISTS ix_pred_user       ON ml_predictions(user_id);

-- ── 7. RISK RANKINGS (fleet-level prioritisation snapshot) ───────────
CREATE TABLE IF NOT EXISTS risk_rankings (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    turbine_id      BIGINT NOT NULL REFERENCES turbines(id) ON DELETE CASCADE,
    computed_at     TIMESTAMPTZ DEFAULT now(),
    fault_probability DOUBLE PRECISION,
    consequence_weight DOUBLE PRECISION,           -- FMECA component weight
    detectability   DOUBLE PRECISION,
    maintenance_coverage DOUBLE PRECISION DEFAULT 0,
    risk_score      DOUBLE PRECISION,              -- 0..1 (or RPN scaled)
    rpn             DOUBLE PRECISION,              -- risk priority number
    risk_class      risk_class_enum,
    rank            INTEGER,                       -- 1 = highest priority
    rul_days        DOUBLE PRECISION,
    component       TEXT,
    financial_exposure DOUBLE PRECISION,           -- € at risk
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_risk_user_rank ON risk_rankings(user_id, rank);

-- ── 8. ASSET HEALTH CERTIFICATES ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS asset_health_certificates (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    turbine_id      BIGINT NOT NULL REFERENCES turbines(id) ON DELETE CASCADE,
    certificate_ref TEXT UNIQUE,                   -- AHC-2026-0529-013
    issued_at       TIMESTAMPTZ DEFAULT now(),
    valid_until     TIMESTAMPTZ,
    overall_health_score DOUBLE PRECISION,         -- 0..100
    risk_class      risk_class_enum,
    classification  TEXT,                          -- HEALTHY / SERVICEABLE / REQUIRES ATTENTION
    component_scores JSONB,                         -- {"gearbox":52,"generator":88,...}
    rul_estimates   JSONB,                          -- {"gearbox":15,...} days
    narrative       TEXT,                           -- LLM/insurer narrative
    model_version   TEXT,
    data_source     data_source_enum DEFAULT 'CSV_UPLOAD',
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_ahc_turbine ON asset_health_certificates(turbine_id, issued_at DESC);
CREATE INDEX IF NOT EXISTS ix_ahc_user    ON asset_health_certificates(user_id);

-- ── 9. MODEL REGISTRY (versioning + monitoring) ──────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    id              BIGSERIAL PRIMARY KEY,
    user_id         BIGINT REFERENCES users(id) ON DELETE CASCADE,   -- null = global model
    version         TEXT NOT NULL,
    algorithm       TEXT NOT NULL,                 -- xgboost / random_forest
    trained_at      TIMESTAMPTZ DEFAULT now(),
    n_train_rows    INTEGER,
    feature_list    JSONB,
    metrics         JSONB,                          -- {"accuracy":..,"f1_macro":..,"roc_auc":..}
    artifact_path   TEXT,
    is_active       BOOLEAN DEFAULT TRUE
);
CREATE INDEX IF NOT EXISTS ix_model_active ON model_registry(user_id, is_active);

-- ════════════════════════════════════════════════════════════════════
--  TimescaleDB upgrade path (optional, for multi-million-row fleets):
--
--    CREATE EXTENSION IF NOT EXISTS timescaledb;
--    SELECT create_hypertable('scada_timeseries', 'ts',
--                             chunk_time_interval => INTERVAL '7 days',
--                             migrate_data => TRUE);
--    -- then add compression + retention policies:
--    ALTER TABLE scada_timeseries SET (timescaledb.compress,
--          timescaledb.compress_segmentby = 'turbine_id');
--    SELECT add_compression_policy('scada_timeseries', INTERVAL '30 days');
-- ════════════════════════════════════════════════════════════════════
