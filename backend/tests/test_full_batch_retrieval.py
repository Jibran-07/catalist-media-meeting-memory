"""Regression checks across every official question, using actual AMI transcripts.

These check evidence retrieval and resumability, not accuracy of external LLM
answers or access to Neon. No expected answer text is hardcoded in the app.
"""
from __future__ import annotations
import asyncio
import json
from pathlib import Path
from app.parser import parse_meeting
from app.evidence_planner import select_evidence
from scripts import generate_answers

ROOT=Path(__file__).resolve().parents[2]
QUESTIONS=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))['questions']


def _all_team_turns(team):
    turns=[]
    for path in sorted((ROOT/'data/transcripts').glob(f'{team}[a-d].txt')):
        p=parse_meeting(path)
        turns.extend({'meeting_id':p.meeting_id,'turn_index':t.index,
                      'timecode':t.timestamp,'speaker':t.speaker,'text':t.text}
                     for t in p.turns)
    return turns


def test_all_25_questions_have_bounded_multi_meeting_evidence():
    assert len(QUESTIONS)==25
    for q in QUESTIONS:
        original=_all_team_turns(q['team'])
        assert len(original)>40, q['question_id']
        selected=select_evidence(q['question'],original,max_chars=14500,
                                 max_turns=90,max_turn_chars=850)
        assert selected, q['question_id']
        assert len({t['meeting_id'] for t in selected})>=2,q['question_id']
        assert len(selected)<=90,q['question_id']
        assert sum(len(t['text'])+75 for t in selected)<=14500,q['question_id']
        lookup={(t['meeting_id'],t['turn_index']):t['text'] for t in original}
        assert all(t['text'] in lookup[(t['meeting_id'],t['turn_index'])]
                   for t in selected), q['question_id']


def test_q04_combination_and_later_cost_cut_are_both_visible():
    q=next(q for q in QUESTIONS if q['question_id']=='Q04')
    s=select_evidence(q['question'],_all_team_turns('TS3011'),
                      max_chars=14500,max_turns=90,max_turn_chars=850)
    assert any(t['meeting_id']=='TS3011c' and 'batteries and the solar' in t['text']
               for t in s)
    assert any(t['meeting_id']=='TS3011d' and 'solar' in t['text'].lower() and
               any(w in t['text'].lower() for w in ('cost','removed','delete','expensive','cut','twelve'))
               for t in s)


def test_batch_continues_after_unresolved_question_and_saves_progress(tmp_path,monkeypatch):
    from unittest.mock import AsyncMock
    questions=[{'question_id':f'Q{i:02d}','question':f'Question {i} on meeting decisions?',
                'team':'TS3009'} for i in range(1,4)]
    data=tmp_path/'data';data.mkdir()
    (data/'questions.json').write_text(json.dumps({'questions':questions}),encoding='utf-8')
    previous={'question_id':'Q01','answer':'Previously source-checked answer.',
              'citations':[{'meeting_id':'TS3009c','quote':'a source quote with at least six words'},
                           {'meeting_id':'TS3009d','quote':'another exact quote with six or more words'}]}
    (tmp_path/'answers.progress.json').write_text(json.dumps({'answers':[previous]}),encoding='utf-8')
    mock=AsyncMock(side_effect=[
        {'status':'insufficient_evidence','details':['No decisive wording found']},
        {'status':'answered','answer':'Another supported result.',
         'citations':[{'meeting_id':'TS3009c','quote':'first exact quote with six or more words'},
                      {'meeting_id':'TS3009d','quote':'six or more literal original transcript words'}]},
    ])
    monkeypatch.setattr(generate_answers,'ROOT',tmp_path)
    monkeypatch.setattr(generate_answers,'PROGRESS',tmp_path/'answers.progress.json')
    monkeypatch.setattr(generate_answers,'FAILURES',tmp_path/'answers.failures.json')
    monkeypatch.setattr(generate_answers,'model_configured',lambda:True)
    monkeypatch.setattr(generate_answers,'ask',mock)
    try:
        asyncio.run(generate_answers.main())
    except SystemExit as exc:
        assert exc.code==2
    else:
        raise AssertionError('Unresolved Q02 must prevent final answers.json creation')
    saved=json.loads((tmp_path/'answers.progress.json').read_text(encoding='utf-8'))
    assert [a['question_id'] for a in saved['answers']]==['Q01','Q03']
    assert saved['answers'][0]==previous
    assert 'Q02' in saved['failures']
    assert not (tmp_path/'answers.json').exists()
    assert 'Q02' in (tmp_path/'answers.failures.json').read_text()
    assert mock.await_count==2