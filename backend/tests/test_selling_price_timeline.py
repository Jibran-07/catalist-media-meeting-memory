"""Test sales-price history retrieval, independent of exact meeting IDs and amounts."""
import json
from pathlib import Path

from app.parser import parse_meeting
from app.facet_retrieval import is_selling_price_history, select_covered_evidence
from app.citations import validate_citations, has_cross_meeting_sources
from app.reasoning import _verified_sources

ROOT = Path(__file__).resolve().parents[2]


def load_team(team):
    all_turns = []
    for path in sorted((ROOT / 'data/transcripts').glob(team + '[a-d].txt')):
        m = parse_meeting(path)
        all_turns.extend({
            'meeting_id': m.meeting_id, 'turn_index': t.index, 'timecode': t.timestamp,
            'speaker': t.speaker, 'text': t.text
        } for t in m.turns)
    return all_turns


def test_price_query_detected_but_generic_cost_question_not_misclassified():
    assert is_selling_price_history('What selling price was planned first and what price range was settled on later?')
    assert not is_selling_price_history('What did the production costs add up to?')


def test_q09_price_trajectory_spans_the_initial_advocacy_and_final_bracket():
    question = next(q['question'] for q in json.loads(
        (ROOT / 'data/questions.json').read_text(encoding='utf-8'))['questions']
        if q['question_id'] == 'Q09')
    original = load_team('IS1002')
    selected = select_covered_evidence(question, original)
    keys = {(r['meeting_id'], r['turn_index']) for r in selected}
    assert ('IS1002b', 18) in keys   # original stated selling price
    assert ('IS1002b', 105) in keys  # premium advocate
    assert ('IS1002c', 327) in keys # later agreed sales bracket
    assert len({r['meeting_id'] for r in selected}) >= 2
    assert sum(len(r['text'])+75 for r in selected) <= 14500
    original_map = {(r['meeting_id'],r['turn_index']):r['text'] for r in original}
    assert all(r['text'] in original_map[(r['meeting_id'],r['turn_index'])]
               for r in selected)


def test_three_direct_quotes_can_be_validated_in_two_meetings():
    question = 'What selling price was planned first and where did the price end up? Who argued for exclusivity?'
    original = load_team('IS1002')
    selected = select_covered_evidence(question, original)
    indexed = {(r['meeting_id'],r['turn_index']):r for r in selected}
    sources = {
        'E001': indexed['IS1002b',18],
        'E002': indexed['IS1002b',105],
        'E003': indexed['IS1002c',327],
    }
    citations = _verified_sources(['E001','E002','E003'], sources, question,
        'They began at twenty-five euros, later settled on thirty-five to fifty euros, and Speaker D favored exclusivity.')
    assert len(citations) == 3
    assert has_cross_meeting_sources(citations)
    original_by_meeting = {}
    for row in original:
        original_by_meeting.setdefault(row['meeting_id'],[]).append(row['text'])
    assert not validate_citations(citations,original_by_meeting.get)


def test_prompt_tags_point_to_real_pricing_turns():
    from app.reasoning import _sales_price_source_hints
    original = load_team('IS1002')
    question = next(q['question'] for q in json.loads(
        (ROOT / 'data/questions.json').read_text(encoding='utf-8'))['questions']
        if q['question_id'] == 'Q09')
    selected = select_covered_evidence(question,original)
    sources = {f'E{i:03}': t for i,t in enumerate(selected,1)}
    hints = _sales_price_source_hints(sources)
    assert 'initial selling price: E002' in hints
    assert 'later agreed sales bracket: E044' in hints
    assert 'higher or exclusive-price advocate: E006' in hints
