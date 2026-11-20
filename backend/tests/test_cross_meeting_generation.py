"""Regression tests for cross-meeting evidence without a live Groq account."""
import asyncio
import json
from pathlib import Path

from app import reasoning
from app.citations import has_cross_meeting_sources, validate_answers
from scripts import generate_answers


def turn(mid, index, text):
    return {'meeting_id': mid, 'turn_index': index, 'timecode': '00:02:00',
            'speaker': 'B', 'text': text}


OLD = turn('TS3009b', 10, 'We are going to make the back out of titanium because that looks nice.')
NEW = turn('TS3009d', 20, 'We are not using titanium anymore because the costs are far too high.')


def test_one_meeting_is_not_enough():
    assert not has_cross_meeting_sources([{'meeting_id': 'TS3009d', 'quote': NEW['text']}])
    assert has_cross_meeting_sources([
        {'meeting_id': 'TS3009b', 'quote': OLD['text']},
        {'meeting_id': 'TS3009d', 'quote': NEW['text']},
    ])


def test_model_reselects_sources_from_different_meetings(monkeypatch):
    answers = iter([
        {'answer': 'Titanium was removed.', 'source_ids': ['E002'], 'missing_information': ''},
        {'answer': 'They planned titanium, then removed it due to cost.',
         'source_ids': ['E001', 'E002'], 'missing_information': ''},
    ])
    calls = []

    async def fake_model(client, messages):
        calls.append(messages)
        return next(answers)

    monkeypatch.setattr(reasoning, 'model_client', lambda: object())
    monkeypatch.setattr(reasoning, '_model_json', fake_model)
    result = asyncio.run(reasoning.call_model(
        'What changed in the design?', [OLD, NEW], require_cross_meeting=True))
    assert len(calls) == 2
    assert {r['meeting_id'] for r in result['citations']} == {'TS3009b', 'TS3009d'}
    assert 'Answer cites only one meeting' in calls[1][-1]['content']


def test_official_model_requires_evidence_from_multiple_meetings(monkeypatch):
    async def should_not_call(client, messages):
        raise AssertionError('Model should not be called when only one meeting is available')

    monkeypatch.setattr(reasoning, 'model_client', lambda: object())
    monkeypatch.setattr(reasoning, '_model_json', should_not_call)
    result = asyncio.run(reasoning.call_model(
        'What changed?', [NEW], require_cross_meeting=True))
    assert not result['citations']
    assert 'another meeting' in result['missing_information']


def test_checkpoint_keeps_strong_and_backs_up_weak(tmp_path, monkeypatch):
    progress = tmp_path / 'answers.progress.json'
    backup = tmp_path / 'answers.before_cross_meeting_repair.json'
    weak = {'question_id': 'Q01', 'answer': 'One meeting only',
            'citations': [{'meeting_id': 'TS3009d', 'quote': NEW['text']}]}
    strong = {'question_id': 'Q02', 'answer': 'Two meetings',
              'citations': [{'meeting_id': 'TS3009b', 'quote': OLD['text']},
                            {'meeting_id': 'TS3009d', 'quote': NEW['text']}]}
    saved = {'answers': [weak, strong], 'brief': 'A completed brief'}
    progress.write_text(json.dumps(saved), encoding='utf-8')
    monkeypatch.setattr(generate_answers, 'PROGRESS', progress)
    monkeypatch.setattr(generate_answers, 'PRE_REPAIR_BACKUP', backup)
    good, failed, brief = generate_answers.load_checkpoint(
        [{'question_id': 'Q01'}, {'question_id': 'Q02'}])
    assert list(good) == ['Q02']
    assert brief == 'A completed brief'
    assert json.loads(backup.read_text(encoding='utf-8')) == saved
    assert json.loads(progress.read_text(encoding='utf-8')) == saved


def test_final_validator_flags_single_meeting():
    from app.parser import parse_meeting
    root = Path(__file__).resolve().parents[2]
    document = json.loads((root / 'answers.json').read_text(encoding='utf-8'))
    questions = json.loads((root / 'data/questions.json').read_text(encoding='utf-8'))['questions']
    # Reuse a real, mechanically valid source quote, but remove the other meeting.
    document['answers'][0]['citations'] = document['answers'][0]['citations'][:1]
    turns = {p.stem: [t.text for t in parse_meeting(p).turns]
             for p in (root / 'data/transcripts').glob('*.txt')}
    failures = validate_answers(document, questions, turns.get)
    assert any('Q01' in msg and 'different meetings' in msg for msg in failures)