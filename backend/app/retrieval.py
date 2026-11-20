"""PostgreSQL indexed retrieval, with separate evidence windows and exact source turns."""
from __future__ import annotations
import re
from collections import defaultdict

SYNONYMS = {
    'screen':'display lcd touch', 'display':'screen lcd',
    'battery':'power energy lithium solar kinetic',
    'speech':'voice recognition microphone sensor',
    'cost':'price budget total expensive component',
    'material':'case casing plastic wood rubber titanium',
    'lost':'find finder locate beeping alarm',
    'shape':'design form fruit vegetable round',
    'buttons':'button navigation wheel control',
}
STOP = set('a an and are as at be by did do for from had has have how in into is it its of on or our that the their them there these this to was were what when where which who why with would'.split())


def keywords(question: str, limit: int = 14) -> list[str]:
    words = [x for x in re.findall(r"[a-zA-Z][a-zA-Z0-9]+", question.lower()) if x not in STOP and len(x)>2]
    result = list(dict.fromkeys(words))
    for word in words:
        for synonym in SYNONYMS.get(word, '').split():
            if synonym not in result:
                result.append(synonym)
    return result[:limit]


def search_passages(conn, question: str, team: str | None, meeting_ids: list[str] | None = None, limit: int = 24):
    """Several shorter ranked OR queries work better for conversational transcripts than a full conjunctive query."""
    terms = keywords(question)
    if not terms:
        return []
    # Team filter applies before ranking through the meetings index.
    # Use a single OR expression; a limit per meeting avoids one long final meeting dominating.
    tsq = ' OR '.join(terms)
    params = [tsq, team, team, meeting_ids, meeting_ids, tsq, limit * 7]
    with conn.cursor() as cur:
        cur.execute('''SELECT p.meeting_id, m.sequence, p.start_index, p.end_index,
                         ts_rank_cd(p.searchable, websearch_to_tsquery('english', %s)) AS rank,
                         p.content
                       FROM passages p JOIN meetings m ON m.meeting_id = p.meeting_id
                       WHERE (%s::text IS NULL OR m.team_id=%s)
                         AND (%s::text[] IS NULL OR p.meeting_id=ANY(%s::text[]))
                         AND p.searchable @@ websearch_to_tsquery('english', %s)
                       ORDER BY rank DESC LIMIT %s''', params)
        rows=cur.fetchall()
    by_meeting=defaultdict(list)
    for meeting_id, seq, start, end, rank, content in rows:
        by_meeting[meeting_id].append({'meeting_id':meeting_id,'sequence':seq,
            'start_index':start,'end_index':end,'score':float(rank),'content':content})
    # Prefer three diverse windows from each meeting before filling remaining slots.
    selected=[]
    for meeting_id in sorted(by_meeting, key=lambda mid: max(r['score'] for r in by_meeting[mid]), reverse=True):
        for row in by_meeting[meeting_id][:3]:
            if len(selected)<limit: selected.append(row)
    return selected


def expand_turns(conn, passages: list[dict], flank: int = 2, max_turns: int = 230):
    """Return precise speaker turns to support answer citation validation."""
    ranges=defaultdict(set)
    for p in passages:
        ranges[p['meeting_id']].update(range(max(0,p['start_index']-flank),p['end_index']+flank+1))
    if not ranges:
        return []
    out=[]
    with conn.cursor() as cur:
        for mid, indices in ranges.items():
            cur.execute('''SELECT meeting_id, turn_index, timecode, speaker, text
                           FROM transcript_turns WHERE meeting_id=%s
                           AND turn_index=ANY(%s::int[]) ORDER BY turn_index''',(mid, sorted(indices)))
            out += [{'meeting_id':r[0], 'turn_index':r[1], 'timecode':r[2],
                     'speaker':r[3], 'text':r[4]} for r in cur.fetchall()]
    return out[:max_turns]


def turns_for_meeting(conn, meeting_id):
    with conn.cursor() as cur:
        cur.execute('SELECT text FROM transcript_turns WHERE meeting_id=%s ORDER BY turn_index',(meeting_id,))
        result=cur.fetchall()
    return [r[0] for r in result] if result else None

# A second, topic-anchored route complements full-text passage ranking. It is
# especially useful when a decision is reversed near the end of a long meeting.
# It scans just one indexed team (usually four meetings), never the entire corpus.
TOPIC_NOISE = set('''team meeting meetings member someone person speaker first next last earlier later they them
plan planned initially originally remote control design designed case back device product idea
what which when where who whose why how did does were was said say says saying asked ask
propose proposed proposing replacing replace changed change switch switched instead
kept keep end final eventually decision decide agreed agree ended happen happened happen
suggest suggested suggesting reason given having made make making point identify
result figure source combination plus one another them their between throughout
'''.split()) | STOP
# Prefer concrete transcript topics to incidental wording in a question.
DOMAIN_TERMS = set("""titanium plastic rubber aluminium aluminum metal wood material
screen lcd display speech voice microphone recognition battery lithium solar kinetic
energy power buttons button navigation touch wheel finder lost colour color
price prices budget cost costs components charging chip shape shapes fruit
vegetable teletext dvd infrared batteries rechargeable""".split())
TOPIC_DECISION = re.compile(
    r'\b(?:replac\w*|chang\w*|dropp\w*|drop|dump\w*|gone|remov\w*|'
    r'instead|agree\w*|decid\w*|keep|kept|stays?|final\w*|'
    r'cost\w*|cheaper|expensiv\w*|budget|price\w*|'
    r'cannot|can\x27t|don\x27t|wouldn\x27t|but|however)\b', re.I)


def topic_candidates(question: str) -> list[str]:
    """Find content terms without treating the question's narrative as search text."""
    found = []
    for term in re.findall(r'\b[a-z][a-z0-9_]{3,}\b', question.lower()):
        if term not in TOPIC_NOISE and term in DOMAIN_TERMS and not re.fullmatch(r'(?:es|ts|is)\d{4}[a-d]?', term):
            if term not in found:
                found.append(term)
    return found[:10]


def rank_topic_matches(matches: list[dict], term: str,
                       question: str, max_turns: int = 30) -> list[dict]:
    """Spread topic occurrences across meetings, preferring decision-related speech.

    This is a deterministic relevance heuristic, not a claim about the final
    decision. The language model still interprets the selected raw turns.
    """
    if not matches:
        return []
    question_words = set(topic_candidates(question))
    grouped = defaultdict(list)
    for row in matches:
        txt = row['text'].lower()
        lexical = sum(1 for w in question_words if w in txt)
        decision = min(3, len(TOPIC_DECISION.findall(txt)))
        # Account for fragmented dialogue without allowing long turns to dominate.
        score = 3 * lexical + 4 * decision + (2 if term in txt else 0)
        score -= min(3, len(txt) // 750)
        grouped[row['meeting_id']].append((score, row))

    result = []
    groups = sorted(grouped)
    # A final meeting cannot displace all evidence of the earlier proposal.
    # Later meetings receive more slots, while earlier proposals still appear.
    # For two meetings this is approximately 1/3 earlier and 2/3 later.
    weights = [2 ** index for index in range(len(groups))]
    total_weight = sum(weights)
    quotas = [max(3, max_turns * weight // total_weight) for weight in weights]
    remaining = max(0, max_turns - sum(quotas))
    for i in range(len(quotas) - 1, -1, -1):
        if remaining == 0:
            break
        quotas[i] += 1
        remaining -= 1
    for meeting_id, quota in zip(groups, quotas):
        ranked = sorted(grouped[meeting_id], key=lambda pair: (-pair[0], -pair[1]['turn_index']))
        result.extend(row for _, row in ranked[:quota])
    result = sorted(result, key=lambda row: (row['meeting_id'], row['turn_index']))
    return result[:max_turns]


def search_topic_turns(conn, question: str, team: str | None,
                       before_meeting: str | None = None,
                       max_turns: int = 30) -> list[dict]:
    """Retrieve original turns for the rarest central topic in the named team.

    Designed for team-scoped questions. An ILIKE scan over 3-4 meetings is cheap
    enough for the assignment and avoids increasing the prompt to corpus size.
    Do not use as an unfiltered query against all meetings.
    """
    if not team:
        return []
    candidates = topic_candidates(question)
    if not candidates:
        return []
    frequencies = []
    with conn.cursor() as cur:
        for word in candidates:
            cur.execute('''SELECT count(*), count(DISTINCT t.meeting_id)
                           FROM transcript_turns t
                           JOIN meetings m ON m.meeting_id=t.meeting_id
                           WHERE m.team_id=%s AND t.text ILIKE %s
                             AND (%s::text IS NULL OR t.meeting_id < %s)''',
                        (team, f'%{word}%', before_meeting, before_meeting))
            count, meeting_count = cur.fetchone()
            if count >= 2:
                frequencies.append((0 if meeting_count >= 2 else 1, count, -len(word), word))
        if not frequencies:
            return []
        focus = min(frequencies)[-1]
        cur.execute('''SELECT t.meeting_id,t.turn_index,t.timecode,t.speaker,t.text
                       FROM transcript_turns t JOIN meetings m ON m.meeting_id=t.meeting_id
                       WHERE m.team_id=%s AND t.text ILIKE %s
                         AND (%s::text IS NULL OR t.meeting_id < %s)
                       ORDER BY m.sequence, t.turn_index LIMIT 400''',
                    (team, f'%{focus}%', before_meeting, before_meeting))
        matches = [{'meeting_id':r[0], 'turn_index':r[1], 'timecode':r[2],
                    'speaker':r[3], 'text':r[4]} for r in cur.fetchall()]
    return rank_topic_matches(matches, focus, question, max_turns=max_turns)


def turns_for_team(conn, team: str, before_meeting: str | None = None) -> list[dict]:
    """Load one team's ordered turns for local evidence ranking.

    Query uses meetings(team_id,sequence) and turns(meeting_id,turn_index)
    indexes, so a larger corpus does not require a corpus-wide table scan.
    """
    if not re.fullmatch(r'[A-Z]{2}\d{4}', team):
        return []
    with conn.cursor() as cur:
        cur.execute('''SELECT t.meeting_id,t.turn_index,t.timecode,t.speaker,t.text
                       FROM meetings m JOIN transcript_turns t ON t.meeting_id=m.meeting_id
                       WHERE m.team_id=%s AND (%s::text IS NULL OR m.meeting_id < %s)
                       ORDER BY m.sequence,t.turn_index''',
                    (team,before_meeting,before_meeting))
        return [{'meeting_id':r[0], 'turn_index':r[1], 'timecode':r[2],
                 'speaker':r[3], 'text':r[4]} for r in cur.fetchall()]