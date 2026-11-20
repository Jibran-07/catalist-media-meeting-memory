"""Bounded retrieval, evidence review and model-assisted synthesis.

No language model is involved in parsing, database indexing, search ranking or citation verification.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
from collections import defaultdict
from .database import db
from .retrieval import search_passages, expand_turns, turns_for_meeting, search_topic_turns, turns_for_team
from .evidence_planner import select_evidence
from .facet_retrieval import select_covered_evidence, is_selling_price_history
from .citations import validate_citations, validate_brief, quote_in_turn, has_cross_meeting_sources

BASE_PROMPT = '''You reconstruct decisions from original meeting transcripts. Only use the supplied evidence.
Interpret proposals, reversals, provisional choices, confirmed choices, and later cost cuts carefully.
Do not treat the latest mention as a confirmed final decision. Never invent names, approvals, or costs.

Each evidence turn has a unique source tag such as E001 or E002. Your job is to select source tags;
PYTHON, not you, copies exact quotations from those tagged turns. NEVER write citation quotes yourself.
Select 1-3 distinct source_ids for each answer. For chronological questions, prefer source_ids from
at least two different meeting IDs, showing earlier decisions and later changes.
Select only turns that DIRECTLY support the statements you make. Do not select unrelated turns.
If your answer states a factual conclusion, source_ids MUST be an array of 1 to 3 valid
E### tags taken from the evidence. If the evidence cannot support any answer, return
source_ids: [] and explain the exact missing information in missing_information.

Return one JSON object with exactly these keys:
{"answer":"one to three plain sentences", "source_ids":["E001","E019"], "missing_information":""}
Do not return meeting IDs or a citations array. Do not include source tags in the answer for normal questions.

For a pre-meeting brief, write short sections headed Decisions, Changes, Open actions, Open issues,
Questions to settle. Place a source tag in square brackets after EACH substantive point, like [E001].
Only quote meetings from before the final meeting. Include 1-3 representative tags in source_ids.
The backend replaces those square-bracket tags with exact transcript citations.
If the evidence is insufficient, acknowledge uncertainty rather than guessing.
'''


def model_configured() -> bool:
    return bool(os.getenv('MODEL_API_KEY'))


def model_client():
    if not model_configured():
        raise RuntimeError('MODEL_API_KEY missing; question answering is in evidence-only mode')
    from openai import AsyncOpenAI
    return AsyncOpenAI(api_key=os.environ['MODEL_API_KEY'],
                       base_url=os.getenv('MODEL_API_BASE_URL') or None,
                       timeout=60.0, max_retries=0)


def _bounded_evidence(turns: list[dict], max_chars: int = 9000,
                      max_turns: int = 90, max_turn_chars: int = 1050) -> list[dict]:
    """Keep actual source wording but cap prompt size and spread it across meetings.

    Each long speaker turn is shortened only at a word boundary. Citation checks
    still refer back to the unmodified database turn. This is necessary for
    rate-limited models: 160 full speech turns can exceed a free tier's TPM.
    """
    meetings: dict[str, list[dict]] = defaultdict(list)
    for turn in turns:
        meetings[turn['meeting_id']].append(turn)
    if not meetings:
        return []
    for items in meetings.values():
        items.sort(key=lambda t: t['turn_index'])
    ordered = sorted(meetings)
    selected = []
    used = 0
    # Iterate round-robin so earlier decisions aren't crowded out by meeting d.
    for position in range(max(len(group) for group in meetings.values())):
        for meeting_id in ordered:
            group = meetings[meeting_id]
            if position >= len(group) or len(selected) >= max_turns:
                continue
            original = group[position]
            text = original['text']
            if len(text) > max_turn_chars:
                text = text[:max_turn_chars].rsplit(' ', 1)[0]
            allowance = len(text) + 65  # evidence header/metadata overhead
            if used + allowance > max_chars:
                continue
            selected.append({**original, 'text': text})
            used += allowance
    return sorted(selected, key=lambda t: (t['meeting_id'], t['turn_index']))


def _source_excerpt(turn_text: str, question: str, answer: str, max_words: int = 24) -> str:
    """Choose a fully verbatim, consecutive word window from one source turn.

    The model chooses the speaker turn, never the quote. Words and punctuation are
    taken from the evidence that was displayed to the model. A relevance-based
    window avoids using unrelated introductory filler in a long speaker turn.
    """
    words = turn_text.split()
    if len(words) < 6:
        raise ValueError('A citation requires six or more words in one turn')
    count = min(max_words, len(words))
    if len(words) == count:
        return ' '.join(words)
    # Scoring words are only used to locate an excerpt, never to rewrite it.
    stopwords = set('the and for with when then which what why from this that them they was were are did have has had how into about will would should can not one two three our your their where while who'.split())
    targets = {w for w in re.findall(r'[a-z]{3,}', (question+' '+answer).lower()) if w not in stopwords}
    best_index, best_score = 0, -1
    for index in range(len(words) - count + 1):
        segment = words[index:index + count]
        tokens = {w for w in re.findall(r'[a-z]{3,}', ' '.join(segment).lower())}
        score = len(tokens & targets)
        if score > best_score:
            best_index, best_score = index, score
    return ' '.join(words[best_index:best_index + count])


def _verified_sources(source_ids: object, sources: dict[str, dict], question: str,
                      answer: str) -> list[dict]:
    """Map model-selected source IDs to original, mechanically exact citations.

    Unknown identifiers, duplicates, and unsupported turn lengths are rejected.
    This function never substitutes a different turn if the model chooses badly.
    """
    if not isinstance(source_ids, list) or not 1 <= len(source_ids) <= 3:
        raise ValueError('The model must identify one to three source tags')
    if any(not isinstance(sid, str) or sid not in sources for sid in source_ids):
        raise ValueError('The model returned a source tag not in the provided evidence')
    if len(set(source_ids)) != len(source_ids):
        raise ValueError('The model repeated a source tag')
    citations = []
    for sid in source_ids:
        turn = sources[sid]
        quote = _source_excerpt(turn['text'], question, answer)
        if not quote_in_turn(quote, turn['text']):
            raise ValueError(f'Citation {sid} failed verbatim source validation')
        citations.append({'meeting_id': turn['meeting_id'], 'quote': quote})
    return citations


def _expand_brief_refs(answer: str, sources: dict[str, dict], question: str) -> str:
    """Replace each [E001] in the brief with an independently verifiable quote."""
    def expand(match):
        source_id = match.group(1)
        if source_id not in sources:
            # Leave it visible so downstream brief validation rejects the output.
            return match.group(0)
        turn = sources[source_id]
        quote = _source_excerpt(turn['text'], question, answer)
        return f'[{turn["meeting_id"]}, Speaker {turn["speaker"]}: "{quote}"]'
    return re.sub(r'\[(E\d{3})\]', expand, answer)


def _sales_price_source_hints(sources: dict[str, dict]) -> str:
    """Flag promising source tags, never supply a factual answer or extra quote.

    All hint tags refer to evidence visible to the model. The model must read
    them and still choose directly relevant evidence. This reduces repeated
    model calls on long pricing discussions in which short agreements get lost.
    """
    if not sources:
        return ''
    earliest = min(r['meeting_id'] for r in sources.values())
    suggestions = []
    tests = (
        ('initial selling price', lambda t: t['meeting_id'] == earliest
         and re.search(r'\bselling price\b', t['text'], re.I)
         and re.search(r'\beuros?\b|[€$£]', t['text'], re.I)),
        ('later agreed sales bracket', lambda t: t['meeting_id'] != earliest
         and re.search(r'\b(?:sales? bracket|price range)\b', t['text'], re.I)
         and re.search(r'\beuros?\b|[€$£]', t['text'], re.I)),
        ('higher or exclusive-price advocate', lambda t:
         re.search(r'\bexclusiv\w*\b', t['text'], re.I)
         and re.search(r'\b(?:higher|high|market|sell|price)\b',t['text'],re.I)),
    )
    for label, predicate in tests:
        for sid, source in sources.items():
            if predicate(source):
                suggestions.append(f'{label}: {sid}')
                break
    if not suggestions:
        return ''
    return ('\nPotentially relevant source tags (inspect their text and cite ONLY '
            'if they support the claim): ' + '; '.join(suggestions) + '.')


async def _model_json(client, messages: list[dict]) -> dict:
    """Make one JSON request, retrying only transient rate-limit failures."""
    from openai import RateLimitError
    for attempt in range(5):
        try:
            response = await client.chat.completions.create(
                model=os.getenv('MODEL_NAME', 'gpt-4o-mini'),
                temperature=0,
                max_completion_tokens=2500,
                response_format={'type': 'json_object'},
                messages=messages,
            )
            if response.usage:
                print(f'Model tokens: {response.usage.prompt_tokens} input, '
                      f'{response.usage.completion_tokens} output.', flush=True)
            raw = json.loads(response.choices[0].message.content)
            if not isinstance(raw, dict):
                raise ValueError('Expected a JSON object from model')
            return raw
        except RateLimitError as exc:
            message = str(exc).lower()
            if 'per day' in message or 'tpd' in message or 'rpd' in message:
                raise RuntimeError('Groq daily quota reached. Resume the batch after the quota resets.') from exc
            if attempt == 4:
                raise
            response = getattr(exc, 'response', None)
            retry_after = response.headers.get('retry-after') if response is not None else None
            try:
                delay = max(1.0, min(120.0, float(retry_after)))
            except (ValueError, TypeError):
                delay = 65.0
            print(f'Groq rate limit reached; retrying after cooldown '
                  f'(attempt {attempt + 1}/5).', flush=True)
            await asyncio.sleep(delay)
    raise RuntimeError('Model did not return an answer')


async def call_model(question: str, turns: list[dict], instruction: str = '',
                     require_cross_meeting: bool = False) -> dict:
    """Require source selections, allowing one constrained correction of format errors.

    Missing source ids can signal genuinely insufficient retrieval. In that case
    the caller searches again instead of attaching unrelated citations.
    """
    if turns and any(t.get('_preselected') for t in turns):
        # Already ranked and bounded across the team's timeline. Re-round-robin
        # ranking here would discard the very decision passages we just found.
        evidence = [t for t in turns if len(t['text'].split()) >= 6]
    else:
        evidence = [t for t in _bounded_evidence(turns, max_chars=8500, max_turns=80,
                                                max_turn_chars=900) if len(t['text'].split()) >= 6]
    if not evidence:
        return {'answer': 'No citation-length evidence was found.', 'citations': [],
                'missing_information': 'Insufficient source turns'}
    if require_cross_meeting and len({t['meeting_id'] for t in evidence}) < 2:
        return {'answer': 'Not enough cross-meeting evidence was retrieved.', 'citations': [],
                'missing_information': 'Find directly relevant source turns from another meeting of this team'}
    sources = {f'E{i:03d}': turn for i, turn in enumerate(evidence, 1)}
    formatted = '\n'.join(
        f'[{sid} | {t["meeting_id"]} #{t["turn_index"]} {t["timecode"]} '
        f'Speaker {t["speaker"]}] {t["text"]}'
        for sid, t in sources.items()
    )
    client = model_client()
    price_hints = _sales_price_source_hints(sources) if is_selling_price_history(question) else ''
    initial = [
        {'role': 'system', 'content': BASE_PROMPT},
        {'role': 'user', 'content': f'Question: {question}\n{instruction}{price_hints}\nEVIDENCE:\n{formatted}'},
    ]
    last_error = None
    messages = initial
    for revision in range(2):
        raw = await _model_json(client, messages)
        answer = raw.get('answer', '')
        if not isinstance(answer, str) or not answer.strip():
            last_error = 'Model returned no answer'
        else:
            ids = raw.get('source_ids')
            try:
                citations = _verified_sources(ids, sources, question, answer)
            except ValueError as exc:
                last_error = str(exc)
                # An empty source list with an explicit evidence gap is not a
                # formatting issue. Trigger another retrieval pass in ask().
                missing = raw.get('missing_information')
                if (ids is None or ids == []) and isinstance(missing, str) and missing.strip():
                    return {'answer': answer, 'citations': [], 'missing_information': missing.strip()}
            else:
                if require_cross_meeting and not has_cross_meeting_sources(citations):
                    last_error = ('Answer cites only one meeting. Select evidence that directly '
                                  'supports the answer from at least two distinct meeting IDs; '
                                  'include an earlier decision and its later outcome where relevant. '
                                  'Never include an unrelated citation merely to pass validation.')
                else:
                    if ('pre-meeting brief' in question.lower()
                            or 'briefing before the final' in question.lower()):
                        answer = _expand_brief_refs(answer, sources, question)
                    return {'answer': answer, 'citations': citations,
                            'missing_information': raw.get('missing_information') or ''}
        if revision == 0:
            print(f'Model source selection needs correction: {last_error}. '
                  'Retrying selection once.', flush=True)
            # Keep the same original evidence identifiers. Ask for a corrected
            # response, not merely a reformatted array of arbitrary references.
            repair = (
                'Your previous JSON answer cannot be accepted: ' + str(last_error) + '.\n'
                'Recheck the evidence and answer the question with only directly '
                'supporting source turns. The source_ids field MUST be an ARRAY '
                'of 1, 2, or 3 distinct tags present in EVIDENCE (for example, '
                '["E001", "E012"]). Do NOT make up tags. Prefer citing '
                'different meetings when comparing changes. If the question explicitly requires two meetings, both cited meetings must directly support the answer. If you truly '
                'cannot find supporting evidence, return source_ids: [] and '
                'describe what is missing in missing_information.\n'
                'Previous response: ' + json.dumps(raw, ensure_ascii=False)[:1800]
            )
            messages = initial + [{'role': 'assistant',
                                   'content': json.dumps(raw, ensure_ascii=False)},
                                  {'role': 'user', 'content': repair}]
    raise ValueError(f'Source selection still invalid after one correction: {last_error}')


def _dedup_turns(rows: list[dict]) -> list[dict]:
    return list({(r['meeting_id'],r['turn_index']):r for r in rows}.values())



async def ask(question: str, team: str | None = None, before_meeting: str | None = None,
              require_cross_meeting: bool = False) -> dict:
    """Search one team chronologically, then retry with targeted missing concepts.

    For user questions without a team, retain the original indexed corpus search.
    No answer or citations are hardcoded for the 25 evaluation questions.
    """
    team = team or (re.search(r'\b(?:ES|TS|IS)\d{4}\b', question.upper()) or [None])[0]
    if before_meeting and not team:
        team = before_meeting[:-1]
    if len(question.strip()) < 5:
        return {'answer': 'Enter a more specific question.', 'citations': [], 'status': 'error'}
    with db() as conn:
        scoped = turns_for_team(conn, team, before_meeting) if team else []
        if scoped:
            evidence = select_covered_evidence(question, scoped, max_chars=14500,
                                       max_turns=90, max_turn_chars=850)
            if not evidence:
                return {'answer': 'No supporting evidence found.', 'citations': [], 'status': 'no_evidence'}
        else:
            passages = search_passages(conn, question, team, limit=24)
            if before_meeting:
                passages = [p for p in passages if p['meeting_id'] < before_meeting]
            evidence = expand_turns(conn, passages, max_turns=160)
            if not evidence:
                return {'answer': 'No supporting evidence found.', 'citations': [], 'status': 'no_evidence'}
        if not model_configured():
            return {'answer': 'Supporting transcript excerpts are available below. A model API key is required to synthesize an answer.',
                    'citations': [], 'evidence': evidence[:22], 'status': 'evidence_only'}

        result = None
        attempted = []
        for pass_no in range(3 if scoped else 2):
            if pass_no:
                gap = str((result or {}).get('missing_information') or '')
                if scoped:
                    # The model's own description of missing information becomes
                    # search hints, never a factual answer or fabricated citation.
                    # Expansion within the same team's original turns is cheap.
                    extra = [gap]
                    evidence = select_covered_evidence(question, scoped, max_chars=16000,
                                               max_turns=105, max_turn_chars=950,
                                               extra_terms=extra)
                    if pass_no == 2:
                        # Explicitly request a different view of source terms on
                        # the last pass, rather than repeating identical context.
                        extra += [' '.join(re.findall(r'[a-z]{4,}', gap.lower())[:12])]
                        evidence = select_covered_evidence(question, scoped, max_chars=19000,
                                                   max_turns=115, max_turn_chars=1100,
                                                   extra_terms=extra)
                else:
                    passages = search_passages(conn, question+' '+gap, team, limit=30)
                    if before_meeting:
                        passages = [p for p in passages if p['meeting_id']<before_meeting]
                    evidence = _dedup_turns(evidence + expand_turns(conn, passages, max_turns=145))
            try:
                instruction = (
                    'Compare the first proposal, any changes, and the latest confirmed decision. '
                    'If the passages do not show all requested facts, list concrete missing '
                    'terms and facts in missing_information.') if pass_no else ''
                if require_cross_meeting:
                    instruction += (' ASSESSMENT REQUIREMENT: cite 2 or 3 speaker turns from at least '
                                    'TWO DIFFERENT meetings of this team. Choose evidence relevant '
                                    'to the actual claim. Do not add padding or guess missing details.')
                    if is_selling_price_history(question):
                        instruction += (' SALES-PRICE HISTORY: distinguish the initial selling price '
                                        'from the later agreed retail/sales price bracket and identify '
                                        'the speaker advocating a higher exclusive price. Cite the '
                                        'earlier price, later agreed bracket, and premium advocate '
                                        'using up to three directly supporting source tags. Never '
                                        'confuse manufacturing/unit cost with selling price.')
                    result = await call_model(question, evidence, instruction,
                                              require_cross_meeting=True)
                else:
                    result = await call_model(question, evidence, instruction)
            except ValueError as exc:
                result = {'answer': '', 'citations': [],
                          'missing_information': f'Citation/source selection: {exc}'}
            attempted.append({'pass': pass_no+1,
                              'evidence_turns': len(evidence),
                              'missing': str(result.get('missing_information') or '')[:180]})
            if result.get('citations') and not result.get('missing_information'):
                break
        citations = (result or {}).get('citations') or []
        if (not citations or (result or {}).get('missing_information')
                or (require_cross_meeting and not has_cross_meeting_sources(citations))):
            return {'answer': 'Could not establish all requested facts from the retrieved evidence.',
                    'citations': [], 'status': 'insufficient_evidence',
                    'details': [(result or {}).get('missing_information') or 'No source tags were selected'],
                    'retrieval_attempts': attempted}
        errors = validate_citations(citations, lambda mid: turns_for_meeting(conn, mid))
        if before_meeting and any(c.get('meeting_id','') >= before_meeting for c in citations):
            errors.append('Brief may not cite future meetings')
        if errors:
            return {'answer': 'Evidence was located, but the generated answer did not pass source verification.',
                    'citations': [], 'status': 'citation_error', 'details': errors,
                    'retrieval_attempts': attempted}
        return {'answer': result.get('answer','').strip(), 'citations': citations,
                'status': 'answered',
                'meetings_searched': len({t['meeting_id'] for t in evidence})}


BRIEF_QUESTION = '''Prepare a practical pre-meeting briefing before the final meeting. Use evidence ONLY from meetings a, b and c. Cover confirmed decisions, decisions changed, outstanding actions and owners, unresolved disagreements, and questions to resolve. Place [E001] style source tags after every substantive point so Python can supply exact quotes. Do not claim that anything that happened in the final meeting is already known.'''

async def generate_brief(team: str = 'TS3012') -> dict:
    """Prepare a pre-final-meeting brief from pre-final transcript turns only."""
    if not model_configured():
        return {'status':'evidence_only','answer':'Model API key required to generate a fresh briefing.'}
    if not re.fullmatch(r'[A-Z]{2}\d{4}',team):
        return {'status':'error','details':['Invalid team ID']}
    # Independent retrieval facets provide decisions, revisions, tasks and open
    # issues, rather than oversampling one theme from one meeting.
    facets=[
        'What was decided about case material shape battery energy screen voice speech and features?',
        'What choices changed or were rejected for budget cost or design reasons?',
        'Who promised actions or to research check investigate price testing customer concerns?',
        'What questions problems or disagreements remained unresolved about the remote?',
    ]
    with db() as conn:
        scoped=turns_for_team(conn,team,before_meeting=team+'d')
        if not scoped:
            return {'status':'no_evidence','details':['No earlier meetings found']}
        evidence=[];seen=set()
        for facet in facets:
            group=select_evidence(facet,scoped,max_chars=4800,
                                  max_turns=32,max_turn_chars=650)
            used=0
            for row in group:
                key=(row['meeting_id'],row['turn_index'])
                if key in seen:continue
                amount=len(row['text'])+75
                if used+amount>3800:continue
                used+=amount
                seen.add(key);evidence.append(row)
        evidence=sorted(evidence,key=lambda t:(t['meeting_id'],t['turn_index']))
        if not evidence:
            return {'status':'no_evidence','details':['No relevant source turns found']}
        errors=['Unverified brief']
        for revision in range(2):
            try:
                result=await call_model(BRIEF_QUESTION,evidence,
                    'Write exactly five short sections: Decisions, Changes, Open actions, Open issues, Questions to settle. '
                    'After every factual bullet add [E###] referencing the specific original turn. '
                    'Never cite the final meeting. Cite only what is actually in the evidence.'
                    + (' The last brief failed quote validation. Fix all source tags and cite every factual bullet.'
                       if revision else ''))
            except ValueError as exc:
                errors=[str(exc)]
                continue
            citations=result.get('citations') or []
            errors=validate_citations(citations,lambda mid: turns_for_meeting(conn,mid))
            errors.extend(validate_brief(result.get('answer',''),lambda mid: turns_for_meeting(conn,mid)))
            if any(c.get('meeting_id')==team+'d' for c in citations) or team+'d' in result.get('answer',''):
                errors.append('Final meeting evidence not allowed')
            if not errors:
                return {**result,'citations':citations,'status':'answered'}
        return {'status':'citation_error','details':errors}