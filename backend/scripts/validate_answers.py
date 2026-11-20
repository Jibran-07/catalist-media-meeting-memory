"""Validate an answer file against original transcript speaker turns."""
import json
from pathlib import Path
from app.parser import parse_meeting
from app.citations import validate_answers

ROOT=Path(__file__).resolve().parents[2]

def validate(file: Path):
    questions=json.loads((ROOT/'data/questions.json').read_text(encoding='utf-8'))['questions']
    document=json.loads(file.read_text(encoding='utf-8'))
    turns={p.stem:[t.text for t in parse_meeting(p).turns]
           for p in (ROOT/'data/transcripts').glob('*.txt')}
    return validate_answers(document,questions,turns.get)

if __name__=='__main__':
    import sys
    path=Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'answers.json'
    errors=validate(path)
    if errors:
        print('\n'.join(errors));sys.exit(1)
    print('PASS: 25 unique questions, every citation is a six-word exact source quote, and brief is present.')