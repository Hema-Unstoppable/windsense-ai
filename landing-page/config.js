// ─── WindSense AI · Supabase Configuration ──────────────────────────
//
// HOW TO GET YOUR KEYS (5 minutes, free):
//   1. Go to https://supabase.com → "Start your project" → sign up free
//   2. Create a new project (name it "windsense-ai")
//   3. Go to: Project Settings → API
//   4. Copy "Project URL" → paste below as SUPABASE_URL
//   5. Copy "anon / public" key → paste below as SUPABASE_ANON_KEY
//   6. Save this file — auth will work immediately.
//
// ─────────────────────────────────────────────────────────────────────

const SUPABASE_URL      = "https://tegudhuanbwczeoykkxp.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_IrseI0q_z1gv2xso5HBmjQ_JSqanIhv";

// ─── Backend API base ───────────────────────────────────────────────
// Local dev: leave as localhost. Production: set to your Render URL,
// e.g. "https://windsense-api.onrender.com" (no trailing slash).
window.WS_API_ROOT = "http://localhost:8000";

// App settings
const APP_CONFIG = {
  appName:      "WindSense AI",
  dashboardUrl: "dashboard.html",
  loginUrl:     "login.html",
  landingUrl:   "index.html",
};
