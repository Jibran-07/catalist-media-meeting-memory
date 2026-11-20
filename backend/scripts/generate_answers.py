"""Generate all 25 model-based answers with source checking and resumable progress.

An unresolved question is recorded, not mistaken for a validated answer. Other
questions continue so one hard retrieval case doesn't block the batch.
Run from backend: python -m scripts.generate_answers
"""
from __future__ import annotations
import asyncio
import json
import os
from pathlib import Path
from app.reasoning import ask, generate_brief, model_configured
from scripts.validate_answers import validate
from app.citations import has_cross_meeting_sources

ROOT = Path(__file__).resolve().parents[2]
PROGRESS = ROOT / 'answers.progress.json'
FAILURES = ROOT / 'answers.failures.json'
PRE_REPAIR_BACKUP = ROOT / 'answers.before_cross_meeting_repair.json'


def save_json(path: Path, content: dict) -> None:
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(content, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def ordered(completed: dict[str, dict], questions: list[dict]) -> list[dict]:
    return [completed[q['question_id']] for q in questions if q['question_id'] in completed]


def load_checkpoint(questions: list[dict]) -> tuple[dict, dict, str | None]:
    valid = {q['question_id'] for q in questions}
    saved = json.loads(PROGRESS.read_text(encoding='utf-8')) if PROGRESS.exists() else {}
    completed = {}
    weak = []
    for item in saved.get('answers', []):
        qid = item.get('question_id')
        if qid not in valid or not item.get('answer') or not item.get('citations'):
            continue
        if has_cross_meeting_sources(item['citations']):
            completed.setdefault(qid, item)
        else:
            weak.append(qid)
    if weak:
        # Never throw away a successfully source-validated earlier attempt.
        if not PRE_REPAIR_BACKUP.exists():
            save_json(PRE_REPAIR_BACKUP, saved)
        print('Answers requiring cross-meeting evidence:', ', '.join(weak), flush=True)
        print('Original progress backed up to', PRE_REPAIR_BACKUP.name, flush=True)
    failures = saved.get('failures') or {}
    if not isinstance(failures, dict):
        failures = {}
    return completed, {qid: value for qid,value in failures.items() if qid in valid}, saved.get('brief')


def checkpoint(completed: dict, failures: dict, questions: list[dict], brief: str | None = None):
    data = {'answers': ordered(completed,questions), 'failures': failures}
    if brief:
        data['brief'] = brief
    save_json(PROGRESS, data)
    unresolved = [{'question_id': q['question_id'], **failures[q['question_id']]}
                  for q in questions if q['question_id'] in failures]
    if 'brief' in failures:
        unresolved.append({'question_id':'brief', **failures['brief']})
    save_json(FAILURES, {'unresolved_questions': unresolved})


def quota_exhausted(exc: Exception) -> bool:
    s = str(exc).lower()
    return 'daily quota' in s or 'per day' in s or 'tokens per day' in s or 'tpd' in s


async def main():
    if not model_configured():
        raise SystemExit('MODEL_API_KEY is required to generate model-based answers.')
    questions = json.loads((ROOT / 'data/questions.json').read_text(encoding='utf-8'))['questions']
    completed, failures, brief = load_checkpoint(questions)
    if completed:
        print(f'Resuming: {len(completed)} previously saved answers.', flush=True)

    for q in questions:
        qid = q['question_id']
        if qid in completed:
            print('Restored', qid, flush=True)
            continue
        print('Working on', qid, flush=True)
        try:
            result = await ask(q['question'], q['team'], require_cross_meeting=True)
        except Exception as exc:
            failures[qid] = {'status': type(exc).__name__, 'details': [str(exc)[:450]]}
            checkpoint(completed,failures,questions,brief)
            if quota_exhausted(exc):
                raise SystemExit('Daily model quota reached. Completed results are saved; '
                                 'rerun after the quota resets. No answers were lost.') from exc
            print('Unresolved', qid, f'({type(exc).__name__}); continuing.', flush=True)
            continue
        if result.get('status') == 'answered' and not has_cross_meeting_sources(result.get('citations', [])):
            result = {'status': 'insufficient_cross_meeting_evidence',
                      'details': ['At least two directly supporting meetings are required']}
        if result.get('status') != 'answered':
            failures[qid] = {'status': result.get('status','unknown'),
                             'details': result.get('details', ['No details']),
                             'retrieval_attempts': result.get('retrieval_attempts',[])}
            checkpoint(completed,failures,questions,brief)
            print('Unresolved', qid, '('+str(result.get('status'))+'); continuing.', flush=True)
            continue
        completed[qid] = {'question_id': qid, 'answer': result['answer'],
                          'citations': result['citations']}
        failures.pop(qid,None)
        checkpoint(completed,failures,questions,brief)
        print('Answered and saved', qid, flush=True)

    unresolved=[q['question_id'] for q in questions if q['question_id'] not in completed]
    if unresolved:
        print(f'Finished attempting all 25 questions: {len(completed)} answered, '
              f'{len(unresolved)} unresolved.', flush=True)
        print('Unresolved IDs:', ', '.join(unresolved), flush=True)
        print('Details are in answers.failures.json; successful answers are in '
              'answers.progress.json. No incomplete answers.json was written.',flush=True)
        raise SystemExit(2)

    if not brief:
        print('Generating TS3012 pre-meeting brief...',flush=True)
        try:
            result=await generate_brief('TS3012')
        except Exception as exc:
            failures['brief']={'status':type(exc).__name__,'details':[str(exc)[:450]]}
            checkpoint(completed,failures,questions)
            raise SystemExit('The 25 answers are saved, but the brief could not be generated. '
                             'See answers.failures.json.') from exc
        if result.get('status') != 'answered':
            failures['brief']={'status':result.get('status'),
                               'details':result.get('details',['Brief verification failed'])}
            checkpoint(completed,failures,questions)
            raise SystemExit('The 25 answers are saved, but the brief failed verification. '
                             'See answers.failures.json.')
        brief=result['answer']
        failures.pop('brief',None)
        checkpoint(completed,failures,questions,brief)
        print('Brief generated and saved.',flush=True)

    candidate=ROOT/'answers.generated.json'
    save_json(candidate,{'answers':ordered(completed,questions),'brief':brief})
    errors=validate(candidate)
    if errors:
        raise SystemExit('Final source validation failed; answers.json was not replaced:\n'+'\n'.join(errors))
    os.replace(candidate,ROOT/'answers.json')
    checkpoint(completed,{},questions,brief)
    print('PASS: All 25 answers and the brief are in project-root answers.json.',flush=True)

if __name__ == '__main__':
    asyncio.run(main())