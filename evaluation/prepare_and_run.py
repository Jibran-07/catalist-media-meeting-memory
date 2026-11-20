"""Independent questions checked against actual individual turns.

This runs an offline lexical retrieval proxy without a PostgreSQL connection.
It does NOT represent end-to-end model or Neon evaluation.
"""
from collections import Counter, defaultdict
from pathlib import Path
from time import perf_counter
import json
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.parser import parse_meeting, make_passages
from app.citations import quote_in_turn
from app.retrieval import keywords

CURATED=[
('E01','TS3012','Which energy source was chosen for Team TS3012 in its third meeting?','Ordinary batteries.','TS3012c',383),
('E02','TS3012','What casing material did Team TS3012 tentatively settle on in its third meeting?','Rubber.','TS3012c',383),
('E03','ES2006','Which high-end features did Team ES2006 keep when simplifying scroll controls?','Voice recognition and the LCD.','ES2006d',417),
('E04','IS1008','What extra unit cost did the IS1008 marketing speaker mention for an LCD?','Approximately two to three euros.','IS1008d',76),
('E05','IS1009','Which specific component was absent from the IS1009 cost calculation?','The voice recogniser.','IS1009d',241),
('E06','IS1003','Which material was proposed to replace wood for Team IS1003?','Plastic.','IS1003d',431),
('E07','ES2010','Why was a lithium battery considered advantageous for Team ES2010?','It offered longer battery life.','ES2010c',227),
('E08','ES2007','What physical problem did the ES2007 marketing speaker associate with excessive channel changing?','Repetitive strain injury and wrist discomfort.','ES2007b',159),
('E09','TS3010','What control layout was suggested instead of the joystick in the final TS3010 meeting?','Conventional up, down, left and right buttons.','TS3010d',204),
('E10','ES2011','Which uncertainty did the ES2011 project manager promise to check before the next meeting?','Whether television-only requirements excluded DVD functions.','ES2011b',229),
('E11','IS1002','What kind of price position did marketing prefer for a beauty-focused IS1002 remote?','A distinctive premium price, rather than economic competition.','IS1002b',105),
('E12','ES2011','What manufacturing concern was raised about irregular fruit or vegetable shapes?','They would be more expensive to produce.','ES2011d',264),
('E13','TS3012','What missing work did the TS3012 project manager lament after removing the scroll wheel?','Reliable finance or price calculations.','TS3012c',635),
('E14','ES2008','Which practical obstacle did Team ES2008 discuss for finding a lost remote by signal?','Infrared cannot find a unit hidden under a sofa.','ES2008d',46),
('E15','TS3009','Which appearance did one designer propose mimicking with cheaper material on the remote front?','A titanium-like finish.','TS3009c',44),
]


def run():
    start=perf_counter()
    meetings={p.stem:parse_meeting(p) for p in (ROOT/'data/transcripts').glob('*.txt')}
    source_entries=[]
    for id_,team,question,expected,mid,idx in CURATED:
        turn=meetings[mid].turns[idx]
        if len(turn.text.split())<6:raise ValueError(mid)
        quote=' '.join(turn.text.split()[:min(len(turn.text.split()),50)])
        assert quote_in_turn(quote,turn.text)
        source_entries.append({'question_id':id_,'team':team,'question':question,
                               'expected_answer':expected,
                               'ground_truth':{'meeting_id':mid,'turn_index':idx,'speaker':turn.speaker,'quote':quote}})
    (ROOT/'evaluation/evaluation_questions.json').write_text(json.dumps({'questions':source_entries},indent=2,ensure_ascii=False)+'\n')
    indexed=defaultdict(list)
    for meeting in meetings.values():
        for p in make_passages(meeting):
            text=p['content'].lower()
            # Separate tokens for reproducible offline proxy (not the PostgreSQL FTS implementation).
            indexed[meeting.team_id].append((meeting.meeting_id, p['start_index'],p['end_index'],Counter(re.findall(r'[a-z][a-z0-9]+',text))))
    results=[]
    for e in source_entries:
        terms=keywords(e['question'],limit=14)
        ranked=[]
        for mid,start_i,end_i,tokens in indexed[e['team']]:
            matching=sum(1 for term in terms if tokens[term]>0)
            frequency=sum(min(3,tokens[term]) for term in terms)
            score=matching+0.05*frequency
            if score>0:ranked.append((score,mid,start_i,end_i))
        ranked.sort(key=lambda p:(-p[0],p[1],p[2]))
        selected=[]
        used=defaultdict(int)
        for score,mid,start_i,end_i in ranked:
            if used[mid]>=2:continue
            selected.append((mid,start_i,end_i))
            used[mid]+=1
            if len(selected)>=5:break
        ground=e['ground_truth']
        hit=any(mid==ground['meeting_id'] and first<=ground['turn_index']<=last for mid,first,last in selected)
        results.append({'question_id':e['question_id'],'question':e['question'],
                        'expected_answer':e['expected_answer'],
                        'source_meeting':ground['meeting_id'],'source_turn':ground['turn_index'],
                        'retrieval_hit_at_5':hit,
                        'top_five_passages':[{'meeting_id':m,'start_turn':s,'end_turn':t} for m,s,t in selected]})
    correct=sum(r['retrieval_hit_at_5'] for r in results)
    output={'evaluation_type':'Offline lexical retrieval proxy, not full agent answer evaluation',
            'scope':'15 new source-checked questions from the supplied transcripts; source turn found within five returned passages',
            'questions_evaluated':len(results), 'retrieval_hits':correct,
            'retrieval_misses':len(results)-correct,
            'retrieval_recall_at_5':round(correct/len(results),3),
            'model_answers_checked':0,
            'reason_for_limited_evaluation':'Neon database and model API credentials unavailable in local environment; backend model answers were not executed',
            'results':results,
            'elapsed_seconds':round(perf_counter()-start,3)}
    (ROOT/'evaluation/results.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print('Offline retrieval:',correct,'/',len(results),'hits; missing:',[r['question_id'] for r in results if not r['retrieval_hit_at_5']]);print('Seconds:',output['elapsed_seconds'])

if __name__=='__main__':run()
