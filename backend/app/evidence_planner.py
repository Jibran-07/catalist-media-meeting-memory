"""Team-scoped, multi-facet evidence selection for ambiguous meeting history questions.

Never requires an external model, and never uses known question IDs or stored answers.
The retrieval scope is one team, not the whole AMI corpus.
"""
from __future__ import annotations
import math
import re
from collections import Counter, defaultdict

NOISE = set('''team teams meeting meetings member members speaker speakers remote control product
would could should which who whom why whose when where how what after first finally final
later earlier initially originally new old previous person somebody someone something
what which did do does had has have was were are is then the and for in to it of a an
be by as with about our their they them that those we you us there here from on or
question answered reported agreed decision decisions decide decided chose choose chosen
instead went change changed changes switch switched say said came ended end did
which one other another rather both next same their them group about more than once
planned picked first leaned proposed suggestions who when enough soon someone years
couldn could does specified went eventually question opinion give given up another
source combination combined happened happening having added out previously design need transcript excerpts excerpt explicitly specific statement statements whether confirmed confirm confirms showing show shows followup
projects project manager favoured favored time kind particular position four last
sometime reason keeping kept original final different switch use used using came
'''.split())
CONCEPTS = {
    'energy': 'energy power battery batteries solar kinetic lithium dynamo charging rechargeable charger source cell',
    'budget': 'cost costs costing budget price prices total euro euros amount expense expensive cheaper cheap finance financial cut removed remove drop dropped omit omitted kept keep screen lcd voice recognition fifteen twelve eight',
    'display': 'screen lcd display touch touchscreen touch-screen interface visual',
    'voice': 'voice speech recognition recogniser recognizer microphone spoken command commands',
    'case': 'material casing case rubber plastic wood titanium metal aluminum aluminium back shell cover',
    'locator': 'lost lose find finder locate locator beep beeper buzzer alarm infrared sensor missing',
    'battery': 'battery batteries lithium aaa cell rechargeable kinetic solar power energy',
    'shape': 'shape shapes fruit fruits vegetable vegetables colour color curved curve design ergonomic comfort',
    'navigation': 'joystick button buttons knob scroll wheel dial navigat navigation arrows push-button',
    'market': 'market selling price sales exclusive expensive premium consumer buyer buyers target age older elderly young younger sixty eighty fifty grandparents grandfather grandfathers grandmother grandmothers',
    'health': 'health injury repetitive rsi pain wrist ergonomic comfortable strain rubber curves curved',
    'dvd': 'dvd tv television video vcr recorder remote channel channels function functions requirements',
}
# Some broad synonyms are only activated when their concept is explicitly present in the question.
TRIGGERS = {
    'energy': {'energy','power','kinetic','solar','dynamo'},
    'budget': {'cost','costs','costing','budget','price','prices','total','finances','expensive','costed'},
    'display': {'display','lcd','screen','touch','touchscreen'},
    'voice': {'speech','voice','recognition','recogniser','recognizer'},
    'case': {'material','rubber','titanium','plastic','wood','case','casing'},
    'locator': {'lost','find','finder','locate','buzzer','alarm'},
    'battery': {'battery','batteries','lithium','aaa'},
    'shape': {'shape','shapes','fruit','vegetable','colours','colors'},
    'navigation': {'navigating','navigation','joystick','buttons','wheel','dial'},
    'market': {'selling','price','exclusive','target','age','buyers'},
    'health': {'health','injury','pain','complaint','wrist','strain'},
    'dvd': {'television','dvd','video','recorder','requirements'},
}
CONCEPTS = {k: set(v.split()) for k,v in CONCEPTS.items()}
CHRONO = re.compile(r'\b(?:agree\w*|decid\w*|chang\w*|replac\w*|switch\w*|keep|kept|drop\w*|remov\w*|cost\w*|budget|total|instead|but|however|then|final\w*|cancel\w*|cheaper|expensiv\w*|scrap\w*|propos\w*)\b', re.I)
NUMERIC = re.compile(r'(?:\b(?:euro|pounds?|dollars?|percent|price|total|budget|cost)\b|(?:€|\$|\d+[,.]\d+))',re.I)


def norm(text: str) -> str:
    # Preserve words spelled as L_C_D_ / T_V_ / D_V_D_ in raw AMI turns.
    text = re.sub(r'\b(?:[A-Za-z]_)\w*', lambda m: m.group(0).replace('_',''), text)
    return text.lower().replace('touch-screen', 'touchscreen').replace('younger','young').replace('batteries','battery')


def tokens(text: str) -> list[str]:
    return re.findall(r'[a-z]+|\d+(?:\.\d+)?', norm(text))


def query_concepts(question: str) -> tuple[set[str], set[str]]:
    raw=set(tokens(re.sub(r'\b(?:ES|TS|IS)\d{4}[a-d]?\b', '', question, flags=re.I))) - NOISE
    # A generic word like 'source' alone must not activate all possible power words.
    expanded=set()
    for name, activation in TRIGGERS.items():
        if raw & activation:
            expanded |= CONCEPTS[name]
    expanded -= raw
    return raw, expanded


def _turn_score(text: str, direct: set[str], expanded: set[str], rare: dict[str,float],
                numeric: bool, changes: bool) -> tuple[float, set[str]]:
    toks=set(tokens(text)); direct_hits=toks & direct; expanded_hits=toks & expanded
    if not direct_hits and not expanded_hits:
        return 0,set()
    d=sum(min(3.8,rare.get(w,1.5)) for w in direct_hits)
    e=sum(min(2,rare.get(w,1.5)) for w in expanded_hits)
    score = 2.7*d + 1.45*e + min(len(direct_hits),3)*1.4
    if len(direct_hits) >= 2: score += 4
    if changes and CHRONO.search(text): score += 3
    if numeric and NUMERIC.search(text): score += 1.8
    # Penalize long, meandering turns enough that short explicit decisions can rank.
    score /= (1+min(2.0,len(text)/1100)*.32)
    return score, direct_hits | expanded_hits


def select_evidence(question: str, turns: list[dict], max_chars: int = 11200,
                    max_turns: int = 62, max_turn_chars: int = 700,
                    extra_terms: list[str] | None = None) -> list[dict]:
    """Construct a prompt evidence pack with source-quote-preserving snippets.

    Several complementary scoring views are used: direct nouns, expanded topic
    concepts, cost/changes, and number-heavy turns. Each meeting receives a quota
    so outcome meetings cannot hide earlier decisions.
    """
    if not turns: return []
    direct, expanded=query_concepts(question)
    direct = set(direct)
    if extra_terms:
        direct |= set(t for x in extra_terms for t in tokens(x)) - NOISE
    for word in ('team','meeting','meetings','remote','control','design','product','speaker'):
        direct.discard(word)
    indexed=[{**row, 'normalized_tokens': set(tokens(row['text']))} for row in turns]
    frequency=Counter(w for row in indexed for w in row['normalized_tokens'])
    count=max(len(turns),1)
    rare={w: max(.75, math.log(1+(count-frequency[w]+.5)/(frequency[w]+.5))) for w in direct | expanded}
    by_meeting=defaultdict(list)
    for row in indexed:
        score, hits=_turn_score(row['text'],direct,expanded,rare,
                               numeric=True, changes=True)
        # Context often comes across multiple short consecutive speaker turns.
        by_meeting[row['meeting_id']].append((row,score,hits))
    # Collect high-relevance centers from every meeting, but give more space to
    # meetings that actually discuss the subject. Diversify across timestamps.
    grouped=[]
    meetings=sorted(by_meeting)
    for mid in meetings:
        items=by_meeting[mid]
        candidates=[]
        for idx,(row,score,hits) in enumerate(items):
            if score<=1.5: continue
            context=items[max(0,idx-2):min(len(items),idx+3)]
            # Avoid low-value cost references without the question's topic.
            context_terms=set().union(*(h for _,_,h in context))
            local = score + .22*sum(sc for _,sc,_ in context)
            # Higher value if the speaker turn has multiple different concepts.
            local += min(4, len(context_terms))*1.25
            # Slight preference for explicit decisions and numerical statements.
            if CHRONO.search(row['text']):local+=1.6
            if NUMERIC.search(row['text']):local+=.7
            candidates.append((local,idx))
        top=[]
        for sc,idx in sorted(candidates, key=lambda item:(-item[0],-item[1])):
            if all(abs(idx-old)>3 for _,old in top):
                top.append((sc,idx))
            if len(top)>=11:break
        for sc,idx in top:
            grouped.append((sc,mid,idx))
    grouped.sort(key=lambda x:(-x[0], -ord(x[1][-1])))
    # Reserve a minimum from every relevant meeting; choose the rest globally.
    ranked=[]
    for mid in meetings:
        relevant=[x for x in grouped if x[1]==mid]
        ranked.extend(relevant[:2])
    ranked.extend(x for x in grouped if x not in ranked)
    seen=set();output=[];budget=0
    meeting_counts=Counter()
    per_meeting_cap=max(6, math.ceil(max_turns/max(1,len(meetings)))*2)
    for score,mid,idx in ranked:
        items=by_meeting[mid]
        # Center, then immediately adjacent response. Preserve original turns.
        for neighbor in (idx,idx+1,idx-1,idx+2):
            if neighbor<0 or neighbor>=len(items): continue
            row=items[neighbor][0]
            key=(mid,row['turn_index'])
            if key in seen:continue
            if len(output)>=max_turns or meeting_counts[mid]>=per_meeting_cap:break
            text=row['text']
            if len(text)>max_turn_chars:
                text=_relevant_snippet(text,question,max_turn_chars,direct,expanded)
            size=len(text)+75
            if budget+size>max_chars:continue
            seen.add(key)
            output.append({k:v for k,v in row.items() if k!='normalized_tokens'} | {'text':text, '_preselected':True})
            meeting_counts[mid]+=1
            budget+=size
    return sorted(output,key=lambda row:(row['meeting_id'],row['turn_index']))


def _relevant_snippet(text: str,question: str,max_chars: int,direct: set[str],expanded: set[str]) -> str:
    """Extract a contiguous region of a long verbatim speaker turn near its key terms."""
    if len(text)<=max_chars: return text
    words=list(re.finditer(r'\S+',text))
    if not words: return text[:max_chars]
    best=(float('-inf'),0,0)
    for center in range(0,len(words),5):
        start=max(0,center-42);end=min(len(words),center+42)
        candidate=text[words[start].start():words[end-1].end()]
        if len(candidate)>max_chars:
            # Switch to a shorter window around the same center.
            start=max(0,center-28);end=min(len(words),center+28)
            candidate=text[words[start].start():words[end-1].end()]
        sc=len(set(tokens(candidate))&direct)*3+len(set(tokens(candidate))&expanded)
        if CHRONO.search(candidate): sc+=2
        if NUMERIC.search(candidate): sc+=1
        if sc>best[0]:best=(sc,start,end)
    _,s,e=best
    return text[words[s].start():words[e-1].end()]