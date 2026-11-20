"""Evidence coverage regression tests using the unmodified AMI source transcripts.

These tests prove source turns appear inside a bounded model input; they do not
pretend that a Groq model has successfully answered the questions.
"""
from __future__ import annotations

import json
from pathlib import Path

from app.parser import parse_meeting
from app.facet_retrieval import select_covered_evidence
from app.citations import quote_in_turn

ROOT = Path(__file__).resolve().parents[2]
QUESTIONS = {q['question_id']:q for q in json.loads(
    (ROOT/'data/questions.json').read_text(encoding='utf-8'))['questions']}


def _team_turns(team: str) -> list[dict]:
    result=[]
    for path in sorted((ROOT/'data/transcripts').glob(f'{team}[a-d].txt')):
        meeting = parse_meeting(path)
        result.extend({'meeting_id':meeting.meeting_id, 'turn_index':t.index,
                       'timecode':t.timestamp, 'speaker':t.speaker, 'text':t.text}
                      for t in meeting.turns)
    return result


def _check(qid: str, required: set[tuple[str,int]]) -> None:
    q=QUESTIONS[qid]
    original=_team_turns(q['team'])
    evidence=select_covered_evidence(q['question'],original,
                                     max_chars=14500,max_turns=90,max_turn_chars=850)
    ids={(t['meeting_id'],t['turn_index']) for t in evidence}
    assert required <= ids, f'{qid} missing {required-ids}'
    assert len({t['meeting_id'] for t in evidence})>=2
    assert len(evidence)<=90
    assert sum(len(t['text'])+75 for t in evidence)<=14500
    source={(t['meeting_id'],t['turn_index']):t['text'] for t in original}
    for t in evidence:
        original_text=source[(t['meeting_id'],t['turn_index'])]
        assert t['text'] in original_text
        if len(t['text'].split())>=6:
            assert quote_in_turn(t['text'],original_text)


def test_cost_limit_total_and_removed_part_are_retrieved():
    # Final meeting: original limit, cost total, explicit removal and response.
    _check('Q15',{('ES2016d',189),('ES2016d',245),
                  ('ES2016d',264),('ES2016d',265)})


def test_lost_remote_approach_and_marketing_reframe_are_retrieved():
    # Separate sensor/buzzer, simple audible trigger, later marketing positioning.
    _check('Q18',{('ES2014b',103),('ES2014b',106),('ES2014c',140)})


def test_battery_plan_kinetic_idea_and_final_power_choice_are_retrieved():
    _check('Q24',{('ES2002c',10),('ES2002c',268),('ES2002d',101)})


def test_all_25_remain_bounded_and_grounded():
    assert len(QUESTIONS)==25
    for qid in QUESTIONS:
        _check(qid,set())