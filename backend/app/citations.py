"""Mechanical citation checks; an LLM cannot validate its own quotations."""
import re
from collections import Counter


def normalize(s: str) -> str:
    # Dataset: case and whitespace differences ignored. Keep punctuation.
    return ' '.join(s.casefold().split())


def word_count(text: str) -> int:
    return len(text.split())


def quote_in_turn(quote: str, original_turn: str) -> bool:
    return word_count(quote) >= 6 and normalize(quote) in normalize(original_turn)


def validate_citations(citations: list[dict], turn_lookup) -> list[str]:
    """turn_lookup(meeting_id) returns iterable of strings for individual turns."""
    errors = []
    if not (1 <= len(citations) <= 3):
        errors.append('Answers must contain one to three citations')
    for i, cite in enumerate(citations):
        meeting_id, quote = cite.get('meeting_id', ''), cite.get('quote', '')
        if not re.fullmatch(r'[A-Z]{2}\d{4}[a-d]', meeting_id):
            errors.append(f'Citation {i + 1}: invalid meeting ID')
            continue
        turns = turn_lookup(meeting_id)
        if turns is None:
            errors.append(f'Citation {i + 1}: unknown meeting ID {meeting_id}')
        elif not any(quote_in_turn(quote, text) for text in turns):
            errors.append(f'Citation {i + 1}: quotation not present in a single speaker turn or under six words')
    return errors


def validate_brief(brief: str, turn_lookup) -> list[str]:
    """Check source quotations used in the included TS3012 pre-meeting briefing."""
    citations = re.findall(r'\[(TS3012[abcd]), Speaker ([ABCD]): "(.*?)"\]', brief, flags=re.S)
    errors = []
    if not citations:
        errors.append('TS3012 brief contains no recognizable inline citations')
    if brief.count('[TS3012') != len(citations):
        errors.append('TS3012 brief contains an unparseable inline citation')
    for meeting_id, speaker, quote in citations:
        if meeting_id == 'TS3012d':
            errors.append('TS3012 brief cites the final meeting')
        turns = turn_lookup(meeting_id)
        if turns is None or not any(quote_in_turn(quote, text) for text in turns):
            errors.append(f'TS3012 brief: invalid quotation from {meeting_id}')
    return errors


def has_cross_meeting_sources(citations: list[dict]) -> bool:
    """At least two distinct cited meetings, not merely two speaker turns."""
    return len({cite.get('meeting_id') for cite in citations}) >= 2


def validate_answers(document: dict, questions: list[dict], turn_lookup) -> list[str]:
    errors = []
    answers = document.get('answers')
    if not isinstance(answers, list):
        return ['Missing answers array']
    required = [q['question_id'] for q in questions]
    supplied = [a.get('question_id') for a in answers]
    if Counter(required) != Counter(supplied):
        errors.append('Question IDs are missing, duplicated, or unexpected')
    if not isinstance(document.get('brief'), str) or not document['brief'].strip():
        errors.append('Missing nonempty brief')
    else:
        errors.extend(validate_brief(document['brief'], turn_lookup))
    for answer in answers:
        id_ = answer.get('question_id', '?')
        if not isinstance(answer.get('answer'), str) or not answer['answer'].strip():
            errors.append(f'{id_}: empty answer')
        citations = answer.get('citations', [])
        errors.extend(f'{id_}: {msg}' for msg in validate_citations(citations, turn_lookup))
        if not has_cross_meeting_sources(citations):
            errors.append(f'{id_}: needs supporting citations from at least two different meetings')
    return errors