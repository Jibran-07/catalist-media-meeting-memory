"""Parse the source files without rewriting or merging speaker turns."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re

HEADER = re.compile(r'^Meeting: ([A-Z]{2}\d{4}[a-d])$', re.M)
TURN = re.compile(r'(?m)^(\d{2}:\d{2}:\d{2}) Speaker ([ABCD])\s*\n')

@dataclass(frozen=True)
class Turn:
    index: int
    timestamp: str
    speaker: str
    text: str

@dataclass(frozen=True)
class Meeting:
    meeting_id: str
    team_id: str
    sequence: int
    turns: list[Turn]


def parse_meeting(path: Path) -> Meeting:
    contents = path.read_text(encoding='utf-8-sig')
    header = HEADER.search(contents)
    if not header or header.group(1) != path.stem:
        raise ValueError(f'Unexpected meeting header: {path}')
    matches = list(TURN.finditer(contents))
    if not matches:
        raise ValueError(f'No speaker turns: {path}')
    turns = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(contents)
        # Only trim surrounding newlines; internal punctuation/spacing is preserved.
        text = contents[match.end():end].strip('\r\n')
        turns.append(Turn(index, match.group(1), match.group(2), text))
    return Meeting(path.stem, path.stem[:-1], 'abcd'.index(path.stem[-1]) + 1, turns)


def make_passages(meeting: Meeting, window: int = 8, step: int = 4) -> list[dict]:
    """Overlapping retrieval passages. Direct quotes still point to original turns."""
    passages = []
    for start in range(0, len(meeting.turns), step):
        group = meeting.turns[start:start + window]
        if not group:
            break
        passages.append({
            'start_index': group[0].index,
            'end_index': group[-1].index,
            'content': '\n'.join(f'Speaker {t.speaker}: {t.text}' for t in group),
        })
    return passages
