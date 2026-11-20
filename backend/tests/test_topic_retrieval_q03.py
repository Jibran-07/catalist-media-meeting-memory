"""Regression: retrieve both the proposal and the cost-driven reversal in Q03.

Uses the original AMI transcript files, not simulated quotes or a model.
"""
import json
from pathlib import Path

from app.parser import parse_meeting
from app.retrieval import topic_candidates, rank_topic_matches
from app.reasoning import _bounded_evidence

ROOT = Path(__file__).resolve().parents[2]


def test_q03_topic_selection_includes_early_proposal_and_final_decision():
    questions = json.loads((ROOT / 'data/questions.json').read_text())['questions']
    question = next(q['question'] for q in questions if q['question_id'] == 'Q03')
    assert topic_candidates(question) == ['titanium']
    matches = []
    for suffix in 'abcd':
        parsed = parse_meeting(ROOT / 'data/transcripts' / f'TS3009{suffix}.txt')
        for turn in parsed.turns:
            if 'titanium' in turn.text.lower():
                matches.append({'meeting_id': parsed.meeting_id, 'turn_index': turn.index,
                                'timecode': turn.timestamp, 'speaker': turn.speaker,
                                'text': turn.text})
    selected = rank_topic_matches(matches, 'titanium', question)
    bounded = _bounded_evidence(selected, max_chars=8500, max_turns=80, max_turn_chars=900)
    indices = {(t['meeting_id'], t['turn_index']) for t in bounded}
    # Speaker B suggests the substitution across two interrupted turns. Later,
    # Speaker C and A confirm titanium was dropped for cost reasons.
    assert {('TS3009d', 363), ('TS3009d', 366), ('TS3009d', 419),
            ('TS3009d', 449)} <= indices
    assert {'TS3009c', 'TS3009d'} <= {t['meeting_id'] for t in bounded}
    assert sum(len(t['text']) + 65 for t in bounded) <= 8500


def test_topic_selection_does_not_search_when_question_has_no_concrete_subject():
    assert topic_candidates('Who first suggested replacing it, and was the change kept?') == []


def test_focused_fallback_uses_real_q03_turns_without_third_generic_call(monkeypatch):
    import asyncio
    from contextlib import contextmanager
    from app import reasoning

    question = next(q['question'] for q in
                    json.loads((ROOT / 'data/questions.json').read_text())['questions']
                    if q['question_id'] == 'Q03')
    raw = {}
    matches = []
    for suffix in 'cd':
        parsed = parse_meeting(ROOT / 'data/transcripts' / f'TS3009{suffix}.txt')
        raw[parsed.meeting_id] = [t.text for t in parsed.turns]
        for t in parsed.turns:
            if 'titanium' in t.text.lower():
                matches.append({'meeting_id': parsed.meeting_id, 'turn_index': t.index,
                                'timecode': t.timestamp, 'speaker': t.speaker,
                                'text': t.text})
    focused = rank_topic_matches(matches, 'titanium', question)

    @contextmanager
    def fake_db():
        yield object()

    generic_queries = []
    model_calls = []

    def fake_search(conn, query, team, limit):
        generic_queries.append(query)
        return [{'meeting_id': 'TS3009d', 'start_index': 0, 'end_index': 8}]

    async def fake_model(q, evidence, instruction=''):
        model_calls.append(evidence)
        if len(model_calls) == 1:
            return {'answer': 'Insufficient evidence', 'citations': [],
                    'missing_information': 'Need the original proposal and final titanium decision'}
        # The model is simulated, but the citations are actual verbatim source turns.
        return {'answer': 'Speaker B proposed titanium-coloured plastic; the team ultimately removed titanium.',
                'citations': [
                    {'meeting_id': 'TS3009d', 'quote': raw['TS3009d'][363]},
                    {'meeting_id': 'TS3009d', 'quote': raw['TS3009d'][366]},
                    {'meeting_id': 'TS3009d', 'quote': raw['TS3009d'][449][:135].rsplit(' ', 1)[0]},
                ], 'missing_information': ''}

    monkeypatch.setattr(reasoning, 'db', fake_db)
    monkeypatch.setattr(reasoning, 'model_configured', lambda: True)
    monkeypatch.setattr(reasoning, 'search_passages', fake_search)
    # Model now searches a bounded team timeline before any corpus-wide query.
    team_turns = []
    for suffix in 'abcd':
        parsed = parse_meeting(ROOT / 'data/transcripts' / f'TS3009{suffix}.txt')
        team_turns.extend({'meeting_id': parsed.meeting_id, 'turn_index': t.index,
                           'timecode': t.timestamp, 'speaker': t.speaker, 'text': t.text}
                          for t in parsed.turns)
    monkeypatch.setattr(reasoning, 'turns_for_team', lambda *a, **k: team_turns)
    monkeypatch.setattr(reasoning, 'expand_turns', lambda *a, **k: focused[:1])
    monkeypatch.setattr(reasoning, 'search_topic_turns', lambda *a, **k: focused)
    monkeypatch.setattr(reasoning, 'call_model', fake_model)
    monkeypatch.setattr(reasoning, 'turns_for_meeting', lambda conn, mid: raw.get(mid))

    result = asyncio.run(reasoning.ask(question, 'TS3009'))
    assert result['status'] == 'answered', result
    assert len(result['citations']) == 3
    assert len(model_calls) == 2
    assert len(generic_queries) == 0
    assert any(t['meeting_id']=='TS3009d' and t['turn_index']==366 for t in model_calls[1])