"""
Create all database tables (portable: SQLite or PostgreSQL).

Run:  python -m backend.scripts.init_db
"""
from __future__ import annotations

from config import Base, engine, settings
import models  # noqa: F401  (registers all ORM tables on Base)


def main():
    print(f"[init_db] target: {'SQLite' if settings.is_sqlite else 'PostgreSQL'}")
    Base.metadata.create_all(bind=engine)
    print("[init_db] all tables created.")


if __name__ == "__main__":
    main()
