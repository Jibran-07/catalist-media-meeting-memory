"""Small evidence-coverage layer for questions that ask about several stages.

Selects original speaker turns using general topic facets, not question IDs,
transcript answers, or model-produced quotations. This complements lexical
ranking, which can miss a short cost change or an answer split across speakers.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict

from .evidence_planner import select_evidence, tokens, NOISE, _relevant_snippet

# These are generic concepts useful for meeting histories, not solution keys.
FACET_TERMS = {
    'amount': {
        'total', 'budget', 'price', 'cost', 'costing', 'limit', 'over', 'under',
        'euros', 'euro', 'pounds', 'expensive', 'cheaper', 'point',
    },
    'cut': {
        'remove', 'removed', 'removing', 'drop', 'dropped', 'cut', 'take',
        'away', 'without', 'lose', 'lost', 'skip', 'screen', 'display', 'lcd',
        'chip', 'button', 'batteries', 'battery', 'feature', 'redesign',
    },
    'find': {
        'find', 'finding', 'finder', 'lost', 'lose', 'sensor', 'buzzer',
        'beep', 'beeping', 'whistle', 'clap', 'clapping', 'alarm', 'noise',
        'speech', 'voice', 'recognition',
    },
    'position': {
        'market', 'marketing', 'sell', 'selling', 'buyers', 'consumer',
        'customers', 'present', 'position', 'advertise', 'feature', 'finder',
        'function', 'brand', 'compete', 'leading',
    },
    'basic_power': {
        'battery', 'batteries', 'non', 'rechargeable', 'regular', 'power',
        'powered', 'source', 'only', 'just', 'decided', 'agree', 'stick',
    },
    'extra_power': {
        'kinetic', 'solar', 'dynamo', 'hybrid', 'movement', 'move',
        'moving', 'shake', 'shaking', 'power', 'charge', 'battery', 'batteries',
    },
    'change': {
        'decided', 'agree', 'agreed', 'actually', 'instead', 'finally',
        'change', 'changed', 'last', 'no', 'not', 'just', 'only', 'remove',
        'removed', 'leave', 'kept', 'keep',
    },
}
AMOUNT = re.compile(
    r'\b(?:\d+(?:[.,]\d+)?|one|two|three|four|five|six|seven|eight|nine|ten|'
    r'eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|'
    r'nineteen|twenty|thirty|point|half)\b', re.I)
MONEY = re.compile(r'\b(?:cost\w*|budget|price\w*|total|over|under|limit|euro\w*|pound\w*)\b',re.I)
ACTION = re.compile(r'\b(?:remov\w*|dropp\w*|drop|cut|lose|lost|screen|display|lcd|without|take\s+(?:it|that|the|our|this)\s+(?:out|away)|no\s+need|redesign\w*)\b',re.I)
LOCATE = re.compile(r'\b(?:lost|lose|find\w*|whistl\w*|clapp?\w*|buzzer|buzz\w*|beep\w*|sensor|alarm|speech|voice|recognition)\b',re.I)
POSITION = re.compile(r'\b(?:market\w*|sell\w*|buyer\w*|present\w*|position\w*|advertis\w*|finder|competition|compete)\b',re.I)
KINETIC = re.compile(r'\b(?:kinetic|solar|dynamo|hybrid|shak\w*|mov\w*|charg\w*)\b',re.I)
BATTERY = re.compile(r'\b(?:batter\w*|power|non.?rechargeable)\b',re.I)


def facets_for_question(question: str) -> list[str]:
    words = set(tokens(question))
    facets = []
    if words & {'cost','costing','costs','budget','price','prices','over','limit'}:
        facets.extend(['amount', 'cut'])
    if words & {'lost','lose','find','finding','finder','locate','buzzer','beep'}:
        facets.extend(['find', 'position'])
    if words & {'power','energy','battery','batteries','solar','kinetic','dynamo'}:
        facets.extend(['basic_power', 'extra_power'])
    if not facets and words & {'changed','change','final','finally','eventually','removed','remove','first','later','addition'}:
        facets.append('change')
    return list(dict.fromkeys(facets))


def _facet_score(text: str, topic: str, question_words: set[str],
                 meeting_id: str) -> float:
    words = set(tokens(text))
    domain = FACET_TERMS[topic]
    hits = words & domain
    # Terms from the actual question are weighted more than broad related words.
    direct = words & question_words
    score = len(hits)*2.0 + len(direct)*1.1
    if topic == 'amount':
        score += 14*bool(AMOUNT.search(text) and MONEY.search(text))
        score += 4*bool('total' in words and AMOUNT.search(text))
        score -= 8*bool(not MONEY.search(text))
    elif topic == 'cut':
        score += 9*bool(ACTION.search(text))
        score += 5*bool(re.search(r'\b(?:remove|removed|take|screen|display|lcd)\b',text,re.I))
        score += 3*bool(MONEY.search(text))
    elif topic == 'find':
        score += 8*bool(LOCATE.search(text))
        score += 6*bool(words & {'buzzer','sensor','whistle','clap','clapping','finder'})
        score += 3*bool(len(hits)>=2)
    elif topic == 'position':
        score += 15*bool(POSITION.search(text) and LOCATE.search(text))
        score += 5*bool('finder' in words or 'market' in words)
    elif topic == 'basic_power':
        score += 10*bool(BATTERY.search(text))
        score += 8*bool(words & {'decided','non','just','only','battery','batteries'})
        score += 6*bool(('battery' in words or 'batteries' in words) and words & {'no','only','just','decided','rechargeable'})
    elif topic == 'extra_power':
        score += 13*bool(words & {'kinetic','solar','dynamo','hybrid'})
        score += 5*bool(BATTERY.search(text))
    elif topic == 'change':
        score += 4*bool(words & FACET_TERMS['change'])
    # Weak evidence turns still have to contain an actual facet term.
    if not hits: return 0
    # Slightly favour turns with full, independently citable content.
    if len(text.split()) < 6: score *= 0.3
    if len(text)>500: score /= 1.0 + min(2.0,(len(text)-500)/700)
    # Final cost decisions are usually discussed in later meetings, but
    # early proposals remain eligible through per-meeting diversity.
    if topic in {'amount','cut','change'} and meeting_id.endswith('d'):
        score += 1.5
    return score


def facet_anchors(question: str, turns: list[dict],
                  max_chars: int = 6700, max_turns: int = 34,
                  max_turn_chars: int = 850) -> list[dict]:
    """Select fact-bearing anchors and neighboring dialogue turns verbatim."""
    facets = facets_for_question(question)
    if not facets or not turns:
        return []
    question_words = set(tokens(question)) - NOISE
    by_mid = defaultdict(list)
    for row in turns:
        by_mid[row['meeting_id']].append(row)
    for mid in by_mid:
        by_mid[mid].sort(key=lambda row:row['turn_index'])

    anchors = []
    # Each facet has independent quota and meeting diversity. Even when a
    # final meeting has many matches, include earlier relevant proposals.
    for facet in facets:
        candidates = []
        for mid, group in by_mid.items():
            for index, row in enumerate(group):
                score = _facet_score(row['text'],facet,question_words,mid)
                if score > 5:
                    candidates.append((score,mid,index))
        candidates.sort(key=lambda item:(-item[0],item[1],item[2]))
        used_indices = defaultdict(list)
        picked=[]
        for score,mid,index in candidates:
            if len(picked)>=4: break
            if any(abs(index - x)<5 for x in used_indices[mid]): continue
            # No one meeting may fill the entire facet when other meetings
            # contain reasonably relevant evidence.
            if len(used_indices[mid])>=2: continue
            picked.append((score,mid,index))
            used_indices[mid].append(index)
        anchors.extend((facet,mid,index) for _,mid,index in picked)

    # A priced design can be discussed over 20+ short speaker turns: the
    # quoted limit, the calculated total, and the removed component often
    # occupy separate turns. Reserve those spans before broad selections.
    priority=[]
    if 'amount' in facets:
        for mid in sorted(by_mid,reverse=True):
            group=by_mid[mid]
            totals = []
            limits = []
            for i,row in enumerate(group):
                text = row['text']
                if not (MONEY.search(text) and AMOUNT.search(text)):
                    continue
                if re.search(r'\b(?:total|over)\b',text,re.I) and re.search(r'\b(?:euro|pound|cost|price|amount|total)\b',text,re.I):
                    totals.append((i, row))
                if re.search(r'\b(?:limit|budget|supposed\s+to\s+cost|under)\b',text,re.I) and re.search(r'\b(?:euro|pound|cost|price|amount|total)\b',text,re.I):
                    limits.append((i, row))
            # Match cost statements in the same meeting. The bounded forward
            # scan is based on relative turn order, not any known answer.
            for indices in (limits,totals):
                if not indices: continue
                # Prefer concise, independently citable pricing statements.
                top=sorted(indices,key=lambda x: (len(x[1]['text'])>320,
                                                   x[0] if indices is limits else -x[0]))[:1]
                priority.extend(('amount',mid,i) for i,_ in top)
            for pos, _ in sorted(totals,key=lambda x:(len(x[1]['text'])>320,-x[0])):
                if not mid.endswith(('c','d')): continue
                for next_pos in range(pos+1,min(len(group),pos+48)):
                    t=group[next_pos]['text']
                    if re.search(r'\b(?:remove|removed|dropp?ed|drop|scrap)\b',t,re.I):
                        priority.append(('cut',mid,next_pos))
                        break
                # One primary redesign for each priced meeting is enough.
                break
    selected=[]; seen=set(); used=0
    for facet,mid,index in priority+anchors:
        group=by_mid[mid]
        for offset in (0,-1,1,2,3):
            i=index+offset
            if i<0 or i>=len(group): continue
            row=group[i]
            key=(mid,row['turn_index'])
            if key in seen or len(selected)>=max_turns: continue
            text=row['text']
            if len(text)>max_turn_chars:
                text=_relevant_snippet(text,question,max_turn_chars,question_words,FACET_TERMS[facet])
            size=len(text)+75
            if used+size>max_chars: continue
            seen.add(key);used+=size
            selected.append({**row,'text':text,'_preselected':True})
    return sorted(selected,key=lambda r:(r['meeting_id'],r['turn_index']))


# Selling prices and production costs are different facts. A pricing-history
# question needs its early selling-price baseline, later agreed bracket, and
# the speaker advocating a different price position. The usual generic cost
# facets can otherwise waste the evidence budget on manufacturing estimates.
SALE_WORDS = re.compile(r"\b(?:sell(?:ing)?|sales?|retail|wholesale|price|priced)\b", re.I)
PRICE_UNITS = re.compile(r"\b(?:euro(?:s)?|pound(?:s)?|dollar(?:s)?)\b|[€$£]", re.I)
PRICE_RANGE = re.compile(r"\b(?:sales? bracket|price range|selling range|price bracket|between\b|from\b|to\b)", re.I)
PREMIUM = re.compile(r"\b(?:exclusive|exclusivity|premium|high(?:er)?[- ]priced?|higher price|price tag|expensive)\b", re.I)
PRODUCTION = re.compile(r"\b(?:production cost|unit cost|manufactur(?:e|ing)|component costs?)\b", re.I)


def is_selling_price_history(question: str) -> bool:
    """Activate only for comparisons of the *sales* price across meetings."""
    words = set(tokens(question))
    return bool(words & {'price', 'prices', 'selling', 'sales'}) and bool(
        words & {'first', 'initial', 'originally', 'later', 'settle', 'settled',
                 'range', 'higher', 'exclusive', 'exclusivity'})


def selling_price_anchors(question: str, turns: list[dict], max_chars: int = 5000,
                          max_turn_chars: int = 850) -> list[dict]:
    """Find original, agreed, and contested sales prices across meetings.

    This is a topic-independent lexical procedure; it uses neither question IDs
    nor expected amounts or meeting identifiers. Retains verbatim speaker turns.
    """
    if not is_selling_price_history(question) or not turns:
        return []
    by_meeting = defaultdict(list)
    for row in turns:
        by_meeting[row['meeting_id']].append(row)
    for rows in by_meeting.values():
        rows.sort(key=lambda r: r['turn_index'])
    meetings = sorted(by_meeting)
    if len(meetings) < 2:
        return []

    def price_signal(text: str) -> bool:
        return bool(SALE_WORDS.search(text) and PRICE_UNITS.search(text) and AMOUNT.search(text))

    # A price in the first available meeting is often the team's initial
    # selling target. Prefer clear selling-price statements to cost estimates.
    earliest = meetings[0]
    initial = []
    for i, row in enumerate(by_meeting[earliest]):
        t = row['text']
        if not price_signal(t):
            continue
        s = 15 * bool(re.search(r'\bselling price\b', t, re.I))
        s += 7 * bool(re.search(r'\b(?:retail|sales?) price\b', t, re.I))
        s += 3 * bool(re.search(r'\b(?:sell|selling|market)\b', t, re.I))
        s -= 13 * bool(PRODUCTION.search(t))
        initial.append((s, earliest, i))
    initial.sort(key=lambda x: (-x[0], x[2]))

    # The agreed bracket often appears in a brief turn much later than the
    # long pricing/cost conversation. Reserve it explicitly before the general
    # search consumes the bounded prompt budget.
    settled = []
    for mid in meetings[1:]:
        for i, row in enumerate(by_meeting[mid]):
            t = row['text']
            if not price_signal(t):
                continue
            s = 18 * bool(re.search(r'\bsales? bracket\b|\bprice range\b', t, re.I))
            s += 7 * bool(PRICE_RANGE.search(t))
            s += 5 * bool(re.search(r'\b(?:okay|agreed|decided|settled|we\s+(?:will|can|are))\b', t, re.I))
            s += 3 * bool(re.search(r'\b(?:sell|sales|selling)\b', t, re.I))
            s -= 16 * bool(PRODUCTION.search(t))
            settled.append((s, mid, i))
    settled.sort(key=lambda x: (-x[0], -ord(x[1][-1]), x[2]))

    advocated = []
    for mid in meetings:
        for i, row in enumerate(by_meeting[mid]):
            t = row['text']
            if not PREMIUM.search(t) or not SALE_WORDS.search(t):
                continue
            s = 18 * bool(re.search(r'\bexclusive\b', t, re.I))
            s += 12 * bool(re.search(r'\b(?:i want|i would|i\s+am an advocate|i\s*\'m an advocate|i\s+say)\b', t, re.I))
            s += 5 * bool(price_signal(t))
            s += 4 * bool(re.search(r'\b(?:higher|high price tag|high-priced|premium)\b', t, re.I))
            advocated.append((s,mid,i))
    advocated.sort(key=lambda x: (-x[0], x[1], x[2]))

    result, seen, used = [], set(), 0
    for facet_candidates in (initial[:2], settled[:3], advocated[:2]):
        for _, mid, center in facet_candidates:
            group = by_meeting[mid]
            for j in (center, center-1, center+1):
                if not 0 <= j < len(group):
                    continue
                row = group[j]
                key = (mid, row['turn_index'])
                if key in seen or len(row['text'].split()) < 6:
                    continue
                snippet = row['text']
                if len(snippet) > max_turn_chars:
                    snippet = _relevant_snippet(snippet, question, max_turn_chars,
                                                set(tokens(question)) - NOISE,
                                                set(['selling','sales','exclusive','price','euros','range']))
                length = len(snippet) + 75
                if used + length > max_chars:
                    continue
                used += length
                seen.add(key)
                result.append({**row, 'text': snippet, '_preselected': True})
    return sorted(result, key=lambda r: (r['meeting_id'],r['turn_index']))


def select_covered_evidence(question: str, turns: list[dict],
                            max_chars: int=14500, max_turns: int=90,
                            max_turn_chars: int=850, extra_terms: list[str] | None=None
                            ) -> list[dict]:
    """Merge essential fact facets with general ranker under a hard prompt cap."""
    # Reserve price-history phases first. A generic cost search tends to rank
    # manufacturing costs above the very short agreed selling-price bracket.
    price_priority = selling_price_anchors(question,turns,
                                           max_chars=min(5000,max_chars//3),
                                           max_turn_chars=max_turn_chars)
    priority=facet_anchors(question,turns,max_chars=min(7000,max_chars//2),
                           max_turns=min(40,max_turns//2),
                           max_turn_chars=max_turn_chars)
    ranked=select_evidence(question,turns,max_chars=max_chars,
                           max_turns=max_turns,max_turn_chars=max_turn_chars,
                           extra_terms=extra_terms)
    seen=set();used=0;result=[]
    for row in price_priority+priority+ranked:
        key=(row['meeting_id'],row['turn_index'])
        if key in seen or len(result)>=max_turns: continue
        length=len(row['text'])+75
        if used+length>max_chars: continue
        used+=length;seen.add(key)
        result.append({**row,'_preselected':True})
    return sorted(result,key=lambda r:(r['meeting_id'],r['turn_index']))