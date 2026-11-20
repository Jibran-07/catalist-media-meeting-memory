"""Database access. The frontend never receives the connection URL."""
from __future__ import annotations
from contextlib import contextmanager
from functools import lru_cache
import os

SCHEMA = '''
CREATE TABLE IF NOT EXISTS meetings (
    meeting_id TEXT PRIMARY KEY,
    team_id TEXT NOT NULL,
    sequence SMALLINT NOT NULL CHECK (sequence BETWEEN 1 AND 4),
    content_sha TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS meetings_team_sequence ON meetings(team_id, sequence);
CREATE TABLE IF NOT EXISTS transcript_turns (
    id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    turn_index INT NOT NULL,
    timecode TEXT NOT NULL,
    speaker CHAR(1) NOT NULL,
    text TEXT NOT NULL,
    UNIQUE(meeting_id, turn_index)
);
CREATE INDEX IF NOT EXISTS turns_meeting_order ON transcript_turns(meeting_id, turn_index);
CREATE TABLE IF NOT EXISTS passages (
    id BIGSERIAL PRIMARY KEY,
    meeting_id TEXT NOT NULL REFERENCES meetings(meeting_id) ON DELETE CASCADE,
    start_index INT NOT NULL,
    end_index INT NOT NULL,
    content TEXT NOT NULL,
    searchable TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE(meeting_id, start_index)
);
CREATE INDEX IF NOT EXISTS passages_fts ON passages USING GIN (searchable);
CREATE INDEX IF NOT EXISTS passages_meeting ON passages(meeting_id, start_index);
'''


def get_connection():
    try:
        import psycopg
    except ImportError as exc:
        raise RuntimeError('Install backend requirements to use PostgreSQL') from exc
    url = os.environ.get('DATABASE_URL')
    if not url:
        raise RuntimeError('DATABASE_URL is not configured')
    # Neon URLs include sslmode=require. Never log credentials.
    return psycopg.connect(url, connect_timeout=12)


@contextmanager
def db():
    with get_connection() as con:
        yield con
