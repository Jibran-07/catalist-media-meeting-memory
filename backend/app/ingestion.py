"""Explicit, idempotent, incremental importer for a Neon PostgreSQL database."""
from __future__ import annotations
import hashlib
from pathlib import Path
from time import perf_counter
from .parser import parse_meeting, make_passages
from .database import db, SCHEMA


def ingest(data_dir: Path) -> dict:
    files = sorted(data_dir.glob('*.txt'))
    if not files:
        raise FileNotFoundError(f'No transcript files in {data_dir}')
    start = perf_counter()
    changed, skipped, turns_total, passage_total = 0, 0, 0, 0
    with db() as conn:
        with conn.cursor() as cur:
            cur.execute(SCHEMA)
        for path in files:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            meeting = parse_meeting(path)
            with conn.cursor() as cur:
                cur.execute('SELECT content_sha FROM meetings WHERE meeting_id = %s', (meeting.meeting_id,))
                stored = cur.fetchone()
                if stored and stored[0] == digest:
                    skipped += 1
                    continue
                cur.execute('''INSERT INTO meetings(meeting_id, team_id, sequence, content_sha)
                               VALUES (%s, %s, %s, %s)
                               ON CONFLICT(meeting_id) DO UPDATE SET
                               team_id=EXCLUDED.team_id, sequence=EXCLUDED.sequence,
                               content_sha=EXCLUDED.content_sha, ingested_at=now()''',
                            (meeting.meeting_id, meeting.team_id, meeting.sequence, digest))
                cur.execute('DELETE FROM transcript_turns WHERE meeting_id=%s', (meeting.meeting_id,))
                cur.execute('DELETE FROM passages WHERE meeting_id=%s', (meeting.meeting_id,))
                cur.executemany('''INSERT INTO transcript_turns
                      (meeting_id, turn_index, timecode, speaker, text)
                      VALUES (%s, %s, %s, %s, %s)''',
                    [(meeting.meeting_id,t.index,t.timestamp,t.speaker,t.text) for t in meeting.turns])
                passages = make_passages(meeting)
                cur.executemany('''INSERT INTO passages
                      (meeting_id, start_index, end_index, content) VALUES (%s,%s,%s,%s)''',
                    [(meeting.meeting_id,p['start_index'],p['end_index'],p['content']) for p in passages])
                changed += 1
                turns_total += len(meeting.turns)
                passage_total += len(passages)
            conn.commit()  # a failure only rolls back the current meeting
    return {'meetings_changed':changed,'meetings_unchanged':skipped,
            'turns_added':turns_total,'passages_added':passage_total,
            'elapsed_seconds':round(perf_counter()-start,3)}
