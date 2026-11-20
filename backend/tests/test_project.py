"""Tests run without outside accounts. Database operations are simulated explicitly."""
from contextlib import contextmanager
from pathlib import Path
import json
import os
import shutil

import pytest
from fastapi.testclient import TestClient
from app.parser import parse_meeting, make_passages
from app.citations import normalize, quote_in_turn, validate_citations, validate_answers
from app.retrieval import keywords
from app.main import app, ROOT
from app import ingestion

TRANSCRIPTS = ROOT / 'data' / 'transcripts'
QUESTIONS = json.loads((ROOT / 'data' / 'questions.json').read_text())['questions']
ANSWER_PATH = ROOT / 'answers.json'
ANSWER_FILE = json.loads(ANSWER_PATH.read_text(encoding="utf-8")) if ANSWER_PATH.exists() else None

@pytest.fixture(scope='module')
def all_meetings():
    return {path.stem: parse_meeting(path) for path in TRANSCRIPTS.glob('*.txt')}


def test_expected_real_dataset(all_meetings):
    assert len(all_meetings) == 138
    assert all(1 <= m.sequence <= 4 for m in all_meetings.values())
    assert all(m.turns for m in all_meetings.values())
    assert sum(len(m.turns) for m in all_meetings.values()) > 20000


def test_metadata_and_speakers(all_meetings):
    meeting = all_meetings['TS3012c']
    assert meeting.team_id == 'TS3012' and meeting.sequence == 3
    assert all(t.speaker in 'ABCD' and len(t.timestamp) == 8 for t in meeting.turns)
    assert [t.index for t in meeting.turns] == list(range(len(meeting.turns)))


def test_passages_preserve_speaker_turns(all_meetings):
    meeting = all_meetings['ES2008b']
    passages = make_passages(meeting, window=8, step=4)
    assert passages[0]['start_index'] == 0 and passages[0]['end_index'] == 7
    assert passages[1]['start_index'] == 4
    assert meeting.turns[0].text in passages[0]['content']
    assert len(passages) < len(meeting.turns)


def test_exact_quote_with_case_and_space_variation():
    assert quote_in_turn('This  is  a   real  quote taken from one turn.',
                         'this is a real quote taken from one turn.')


def test_quote_rejects_short_strings():
    assert not quote_in_turn('this quote is too short', 'this quote is too short')


def test_quote_rejects_paraphrase_and_different_punctuation():
    assert not quote_in_turn('it was extremely costly to do this', 'it was too costly to do this')
    assert not quote_in_turn('it was, extremely costly to do this', 'it was extremely costly to do this')


def test_cross_turn_citation_is_invalid():
    original = ['One two three four five six.', 'Seven eight nine ten eleven twelve.']
    errors = validate_citations([{'meeting_id':'ES2008b', 'quote':'five six Seven eight nine ten'}],
                                lambda _: original)
    assert errors


def test_unknown_meeting_is_invalid():
    errors = validate_citations([{'meeting_id':'ZZ1234z', 'quote':'One two three four five six'}], lambda _: None)
    assert errors


def test_official_answers_are_unique_and_valid(all_meetings):
    if ANSWER_FILE is None: pytest.skip('Final answers.json is not generated yet')
    errors = validate_answers(ANSWER_FILE, QUESTIONS,
                              lambda mid: [t.text for t in all_meetings[mid].turns] if mid in all_meetings else None)
    assert errors == []
    assert len(ANSWER_FILE['answers']) == 25


def test_every_answer_has_cross_meeting_sources():
    if ANSWER_FILE is None: pytest.skip('Final answers.json is not generated yet')
    assert all(len({c['meeting_id'] for c in a['citations']}) >= 2 for a in ANSWER_FILE['answers'])


def test_brief_does_not_cite_future_meeting():
    if ANSWER_FILE is None: pytest.skip('Final answers.json is not generated yet')
    assert 'TS3012d' not in ANSWER_FILE['brief']
    for required in ('Decisions', 'Changes', 'Open actions', 'Questions to settle'):
        assert required in ANSWER_FILE['brief']


def test_synonym_query_terms():
    terms = keywords('Why was the speech recognition feature removed due to cost?')
    assert 'speech' in terms
    assert 'voice' in terms
    assert 'recognition' in terms
    assert len(terms) <= 14


def test_health_endpoint_without_database(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    monkeypatch.delenv('MODEL_API_KEY', raising=False)
    response = TestClient(app).get('/health')
    assert response.status_code == 200
    assert response.json()['database'] == 'not_configured'
    assert response.json()['reasoning'] == 'evidence_only'


def test_questions_endpoint():
    response = TestClient(app).get('/api/questions')
    assert response.status_code == 200
    assert len(response.json()['questions']) == 25


def test_answers_endpoint():
    if ANSWER_FILE is None: pytest.skip('Final answers.json is not generated yet')
    response = TestClient(app).get('/api/answers')
    assert response.status_code == 200
    assert len(response.json()['answers']) == 25


def test_brief_endpoint():
    if ANSWER_FILE is None: pytest.skip('Final answers.json is not generated yet')
    response = TestClient(app).get('/api/brief')
    assert response.status_code == 200
    assert 'TS3012' in response.json()['brief']


def test_ask_rejects_too_short_query():
    response=TestClient(app).post('/api/ask', json={'question':'no'})
    assert response.status_code == 422


def test_ask_reports_missing_database(monkeypatch):
    monkeypatch.delenv('DATABASE_URL', raising=False)
    response=TestClient(app).post('/api/ask', json={'question':'Which changes did they make?','team':'TS3012'})
    assert response.status_code == 503
    assert response.json()['detail'] == 'Meeting search is currently unavailable'


class MockCursor:
    def __init__(self, db):
        self.db = db
        self._row = None

    def __enter__(self): return self
    def __exit__(self, *args): return False

    def execute(self, query, params=()):
        q = query.strip().upper()
        if q.startswith('SELECT CONTENT_SHA'):
            key=params[0]
            self._row = (self.db.meetings[key],) if key in self.db.meetings else None
        elif q.startswith('INSERT INTO MEETINGS'):
            self.db.meetings[params[0]] = params[3]
        elif q.startswith('DELETE FROM TRANSCRIPT_TURNS'):
            self.db.turns[params[0]] = []
        elif q.startswith('DELETE FROM PASSAGES'):
            self.db.passages[params[0]] = []
        elif q.startswith('CREATE TABLE'):
            self.db.schema_initialized = True
        else:
            raise AssertionError(f'Unexpected SQL: {q[:100]}')

    def fetchone(self):return self._row

    def executemany(self, query, data):
        q=query.upper()
        if 'INSERT INTO TRANSCRIPT_TURNS' in q:
            for row in data:self.db.turns.setdefault(row[0], []).append(row)
        elif 'INSERT INTO PASSAGES' in q:
            for row in data:self.db.passages.setdefault(row[0], []).append(row)
        else:
            raise AssertionError(query)


class MockConnection:
    def __init__(self):
        self.meetings = {}
        self.turns = {}
        self.passages = {}
        self.schema_initialized = False
        self.commits=0

    def cursor(self):return MockCursor(self)
    def commit(self):self.commits += 1


@contextmanager
def mocked_database(conn):
    yield conn


def test_incremental_ingest_and_safe_rerun(monkeypatch,tmp_path):
    example = TRANSCRIPTS/'ES2008a.txt'
    copied = tmp_path/example.name
    shutil.copy2(example,copied)
    conn=MockConnection()
    monkeypatch.setattr(ingestion,'db',lambda: mocked_database(conn))
    first=ingestion.ingest(tmp_path)
    assert first['meetings_changed']==1 and first['meetings_unchanged']==0
    assert conn.schema_initialized
    assert len(conn.turns['ES2008a']) == len(parse_meeting(example).turns)
    assert len(conn.passages['ES2008a']) > 0
    turns_count=len(conn.turns['ES2008a'])
    again=ingestion.ingest(tmp_path)
    assert again['meetings_changed']==0 and again['meetings_unchanged']==1
    assert len(conn.turns['ES2008a'])==turns_count
    with copied.open('a') as f:f.write('\n')
    third=ingestion.ingest(tmp_path)
    assert third['meetings_changed']==1 and len(conn.turns['ES2008a'])==turns_count


def test_empty_ingestion_dir_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):ingestion.ingest(tmp_path)