"""Q03-style source-id format failures should not throw away completed answers."""
import asyncio
from contextlib import contextmanager

from app import reasoning


def _turn(meeting_id, index, text):
    return {'meeting_id': meeting_id, 'turn_index': index,
            'timecode': '00:01:00', 'speaker': 'B', 'text': text}


FIRST = _turn('TS3009c', 22, 'We thought of titanium as a material and discussed making the back from it originally.')
SECOND = _turn('TS3009d', 61, 'Perhaps instead we could make the back from plastic and give it a titanium appearance.')


def test_invalid_source_count_gets_one_constrained_correction(monkeypatch):
    responses = [
        {'answer': 'The proposal changed.', 'source_ids': [], 'missing_information': ''},
        {'answer': 'The proposal changed from titanium to plastic.',
         'source_ids': ['E001', 'E002'], 'missing_information': ''},
    ]
    calls = []

    async def fake_json(client, messages):
        calls.append(messages)
        return responses.pop(0)

    monkeypatch.setattr(reasoning, 'model_client', lambda: object())
    monkeypatch.setattr(reasoning, '_model_json', fake_json)
    actual = asyncio.run(reasoning.call_model('How did the titanium proposal change?', [FIRST, SECOND]))
    assert len(calls) == 2
    assert 'source_ids field MUST be an ARRAY' in calls[1][-1]['content']
    assert {c['meeting_id'] for c in actual['citations']} == {'TS3009c', 'TS3009d'}
    for citation in actual['citations']:
        assert any(reasoning.quote_in_turn(citation['quote'], t['text'])
                   for t in [FIRST, SECOND] if t['meeting_id'] == citation['meeting_id'])


def test_empty_source_list_and_named_evidence_gap_requests_retrieval(monkeypatch):
    async def fake_json(client, messages):
        return {'answer': 'Not enough evidence yet.', 'source_ids': [],
                'missing_information': 'final costing of titanium back'}

    monkeypatch.setattr(reasoning, 'model_client', lambda: object())
    monkeypatch.setattr(reasoning, '_model_json', fake_json)
    actual = asyncio.run(reasoning.call_model('Was the titanium kept?', [FIRST, SECOND]))
    assert actual['citations'] == []
    assert 'final costing' in actual['missing_information']


def test_invalid_ids_remain_rejected_after_bounded_retry(monkeypatch):
    async def fake_json(client, messages):
        return {'answer': 'Unsupported.', 'source_ids': ['E999'], 'missing_information': ''}

    monkeypatch.setattr(reasoning, 'model_client', lambda: object())
    monkeypatch.setattr(reasoning, '_model_json', fake_json)
    try:
        asyncio.run(reasoning.call_model('What happened?', [FIRST]))
    except ValueError as exc:
        assert 'after one correction' in str(exc)
    else:
        raise AssertionError('Unknown source IDs must not be silently replaced')


def test_gap_first_query_changes_retrieval_terms(monkeypatch):
    @contextmanager
    def fake_db():
        yield object()

    queries = []

    def fake_search(conn, query, team, limit):
        queries.append(query)
        return [{'meeting_id': 'TS3009d', 'start_index': 0, 'end_index': 4}]

    def fake_expand(conn, passages, max_turns):
        return [FIRST, SECOND]

    answers = iter([
        {'answer': 'Needs more evidence.', 'citations': [], 'missing_information': 'final costing titanium plastic'},
        {'answer': 'Titanium was replaced with plastic.',
         'citations': [{'meeting_id': 'TS3009d', 'quote': SECOND['text']}],
         'missing_information': ''},
    ])

    async def fake_model(question, evidence, instruction=''):
        return next(answers)

    monkeypatch.setattr(reasoning, 'model_configured', lambda: True)
    monkeypatch.setattr(reasoning, 'db', fake_db)
    monkeypatch.setattr(reasoning, 'search_passages', fake_search)
    monkeypatch.setattr(reasoning, 'turns_for_team', lambda *a, **k: [])
    monkeypatch.setattr(reasoning, 'expand_turns', fake_expand)
    monkeypatch.setattr(reasoning, 'search_topic_turns', lambda *a, **k: [])
    monkeypatch.setattr(reasoning, 'call_model', fake_model)
    monkeypatch.setattr(reasoning, 'turns_for_meeting', lambda conn, mid: [SECOND['text']])
    actual = asyncio.run(reasoning.ask('Was titanium kept in the end?', 'TS3009'))
    assert actual['status'] == 'answered'
    assert len(queries) == 2
    assert 'final costing titanium plastic' in queries[1]