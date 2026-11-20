"""Regression checks for source-tag citations, using real meeting speech without an API key."""
import asyncio
import json
import sys
from types import ModuleType, SimpleNamespace

import pytest

from app import reasoning
from app.citations import quote_in_turn, validate_brief, validate_citations
from app.parser import parse_meeting
from app.main import ROOT


def _real_turn(meeting_id, search_word=None):
    meeting = parse_meeting(ROOT / 'data' / 'transcripts' / f'{meeting_id}.txt')
    turns = [t for t in meeting.turns if len(t.text.split()) >= 24 and len(t.text) <= 750]
    if search_word:
        turns = [t for t in turns if search_word.lower() in t.text.lower()]
    assert turns, f'No suitable turn in {meeting_id}'
    t = turns[0]
    return {'meeting_id': meeting_id, 'turn_index': t.index,
            'timecode': t.timestamp, 'speaker': t.speaker, 'text': t.text}


def _fake_model(monkeypatch, result):
    # No external API package needed for local regression tests.
    fake_openai = ModuleType('openai')
    fake_openai.RateLimitError = type('RateLimitError', (Exception,), {})
    monkeypatch.setitem(sys.modules, 'openai', fake_openai)
    async def create(**kwargs):
        assert 'E001' in kwargs['messages'][1]['content']
        assert kwargs['response_format'] == {'type': 'json_object'}
        return SimpleNamespace(
            usage=None,
            choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(result)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(reasoning, 'model_client', lambda: client)


def test_verbatim_citations_from_source_tags(monkeypatch):
    turns = [_real_turn('TS3006b'), _real_turn('TS3006d')]
    _fake_model(monkeypatch, {
        'answer': 'A decision changed after further consideration.',
        'source_ids': ['E001', 'E002'],
        'missing_information': '',
        'citations': [{'meeting_id': 'FAKE', 'quote': 'A quotation that was never spoken'}],
    })
    result = asyncio.run(reasoning.call_model('Which decision changed?', turns))
    assert len(result['citations']) == 2
    assert {c['meeting_id'] for c in result['citations']} == {'TS3006b', 'TS3006d'}
    for turn, citation in zip(turns, result['citations']):
        assert quote_in_turn(citation['quote'], turn['text'])
    assert validate_citations(result['citations'],
                              lambda mid: [t['text'] for t in turns if t['meeting_id'] == mid]) == []


def test_invalid_source_tag_is_rejected(monkeypatch):
    turns = [_real_turn('TS3006b')]
    _fake_model(monkeypatch, {'answer': 'This is unsupported.',
                              'source_ids': ['E999'], 'missing_information': ''})
    with pytest.raises(ValueError, match='not in the provided evidence'):
        asyncio.run(reasoning.call_model('What happened?', turns))


def test_generated_premeeting_brief_has_exact_inline_quotes(monkeypatch):
    turns = [_real_turn('TS3012a'), _real_turn('TS3012b'), _real_turn('TS3012c')]
    _fake_model(monkeypatch, {
        'answer': 'Decisions: An earlier item was proposed [E001]. Changes: A later change was discussed [E002]. Open actions: A follow-up is needed [E003].',
        'source_ids': ['E001', 'E002', 'E003'],
        'missing_information': '',
    })
    result = asyncio.run(reasoning.call_model(reasoning.BRIEF_QUESTION, turns))
    assert '[E001]' not in result['answer']
    assert '[TS3012a, Speaker ' in result['answer']
    assert '[TS3012b, Speaker ' in result['answer']
    assert '[TS3012c, Speaker ' in result['answer']
    assert validate_brief(result['answer'],
                          lambda mid: [t['text'] for t in turns if t['meeting_id'] == mid]) == []


def test_excerpt_not_rewritten():
    text = 'One two three four five six seven eight nine ten, eleven twelve thirteen fourteen fifteen sixteen.'
    quote = reasoning._source_excerpt(text, 'Which changes happened?', 'Changed one choice.')
    assert quote_in_turn(quote, text)
    assert len(quote.split()) >= 6