"""
Auth / tenant dependency.

  AUTH_MODE=dev  → single-tenant: every request resolves to the first user
                   (perfect for local development and the pilot demo).
  AUTH_MODE=jwt  → verifies a Supabase JWT (Bearer token), resolves the
                   tenant by auth_uid, auto-provisioning the user row.

Either way, downstream code receives a `user` object and scopes all
queries by user.id — the multi-tenant isolation boundary.
"""
from __future__ import annotations

from fastapi import Depends, Header, HTTPException
from sqlalchemy import select, func
from sqlalchemy.orm import Session

from config import settings, get_db


def get_current_user(
    db: Session = Depends(get_db),
    authorization: str | None = Header(default=None),
    x_ws_user_email: str | None = Header(default=None, alias="X-WS-User-Email"),
):
    from models import User

    # ── email-scoped mode (pilot) ─────────────────────────────────
    # The frontend, after a real Supabase login, sends the user's email.
    # Each email gets its own isolated tenant. (Harden to JWT for prod.)
    if settings.AUTH_MODE == "email_header":
        email = (x_ws_user_email or "").strip().lower()
        if email:
            user = db.execute(
                select(User).where(func.lower(User.email) == email)
            ).scalar()
            if not user:
                # new login with no data yet → create an empty tenant
                user = User(email=email, display_name=email.split("@")[0])
                db.add(user)
                db.commit()
            return user
        # no email header (e.g. direct API/docs call) → fall back to first tenant
        user = db.execute(select(User).order_by(User.id)).scalars().first()
        if not user:
            raise HTTPException(503, "No tenant seeded. Run: python -m scripts.seed_first_user")
        return user

    if settings.AUTH_MODE == "jwt":
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "Missing bearer token")
        token = authorization.split(" ", 1)[1]
        claims = _decode_supabase_jwt(token)
        uid = claims.get("sub")
        email = claims.get("email")

        # 1) already linked by Supabase user id
        user = db.execute(select(User).where(User.auth_uid == uid)).scalar()
        if user:
            return user

        # 2) claim an existing tenant seeded by email (case-insensitive — Supabase lowercases)
        if email:
            user = db.execute(
                select(User).where(func.lower(User.email) == email.strip().lower())
            ).scalar()
            if user:
                user.auth_uid = uid          # link this Supabase identity to the tenant
                db.commit()
                return user

        # 3) brand-new user → create an empty tenant (no farm data yet)
        user = User(auth_uid=uid, email=email or f"{uid}@users.windsense.ai",
                    display_name=(email or uid).split("@")[0])
        db.add(user)
        db.commit()
        return user

    # dev mode — first user
    user = db.execute(select(User).order_by(User.id)).scalars().first()
    if not user:
        raise HTTPException(503, "No user seeded. Run (from backend/): python -m scripts.seed_first_user")
    return user


_JWK_CLIENT = None


def _jwk_client():
    """Cached JWKS client for asymmetric (RS256/ES256) Supabase signing keys."""
    global _JWK_CLIENT
    if _JWK_CLIENT is None:
        from jwt import PyJWKClient
        url = settings.SUPABASE_URL.rstrip("/") + "/auth/v1/.well-known/jwks.json"
        _JWK_CLIENT = PyJWKClient(url)
    return _JWK_CLIENT


def _decode_supabase_jwt(token: str) -> dict:
    """
    Verify a Supabase access token. Supports BOTH signing modes:
      • asymmetric (RS256/ES256) — verified against the project's public JWKS
      • legacy HS256 — verified with SUPABASE_JWT_SECRET
    """
    import jwt
    try:
        alg = jwt.get_unverified_header(token).get("alg", "HS256")
        if alg in ("RS256", "ES256"):
            key = _jwk_client().get_signing_key_from_jwt(token).key
            return jwt.decode(token, key, algorithms=[alg],
                              audience="authenticated", options={"verify_aud": True})
        # HS256 (legacy shared secret)
        if not settings.SUPABASE_JWT_SECRET:
            raise HTTPException(500, "Project uses HS256 but SUPABASE_JWT_SECRET is not set")
        return jwt.decode(token, settings.SUPABASE_JWT_SECRET, algorithms=["HS256"],
                          audience="authenticated")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(401, f"Invalid token: {e}")
