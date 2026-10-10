import os
from pathlib import Path
from typing import Generator
from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker
from app.core.config import settings
from app.db.base import Base


def _ensure_sqlite_directory(db_url: str) -> None:
    """Ensure directory exists for file-based SQLite database URLs."""
    if db_url.startswith("sqlite:///") and not db_url.startswith("sqlite:///:memory:"):
        # Extract file path after sqlite:///
        file_path_str = db_url.replace("sqlite:///", "")
        # Handle relative or absolute paths
        file_path = Path(file_path_str)
        if file_path.parent:
            file_path.parent.mkdir(parents=True, exist_ok=True)


_ensure_sqlite_directory(settings.database_url)

# Connect args: check_same_thread is required False for SQLite in multithreaded FastAPI
connect_args = {}
if settings.database_url.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_engine(
    settings.database_url,
    connect_args=connect_args,
    pool_pre_ping=True,
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Enable SQLite foreign key constraints and WAL mode for high concurrency."""
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        # WAL mode is effective for file-based databases
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()
    except Exception:
        # Ignore errors for drivers or memory DBs where WAL or pragmas are not supported
        pass


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency that provides an isolated SQLAlchemy database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db(target_engine=None) -> None:
    """Create all tables registered with Base.metadata (fallback or test setup)."""
    # Import all models to ensure they are registered with metadata
    import app.db.models  # noqa: F401

    e = target_engine or engine
    Base.metadata.create_all(bind=e)
