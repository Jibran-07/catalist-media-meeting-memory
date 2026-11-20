"""Local ingestion preprocessing benchmark. Does not pretend to measure Neon inserts."""
from time import perf_counter
from pathlib import Path
from statistics import median
import json, sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from app.parser import parse_meeting,make_passages

def measure():
    files=sorted((ROOT/'data/transcripts').glob('*.txt'))
    runs=[]; stats={}
    for repeat in range(5):
        start=perf_counter(); turn_count=0; passage_count=0; source_bytes=0; passage_bytes=0
        for p in files:
            m=parse_meeting(p)
            passages=make_passages(m)
            turn_count+=len(m.turns);passage_count+=len(passages)
            source_bytes+=p.stat().st_size
            passage_bytes+=sum(len(w['content'].encode('utf-8')) for w in passages)
        runs.append(perf_counter()-start)
    median_time=median(runs)
    record={
      'task':'Local Python transcript reading, speaker parsing and overlapping passage construction only',
      'not_measured':'PostgreSQL schema creation, network round trips, managed database writes, GIN index updates, model inference, live backend latency',
      'sample_meetings':len(files),'sample_speaker_turns':turn_count,'sample_passages':passage_count,
      'sample_source_bytes':source_bytes,'sample_overlapping_passage_bytes':passage_bytes,
      'repeat_count':len(runs),'all_runs_seconds':[round(x,4) for x in runs],
      'median_local_preprocessing_seconds':round(median_time,4),
      'estimated_local_preprocessing_seconds_for_1000':round(median_time*1000/len(files),3),
      'estimated_1000_source_megabytes_decimal':round(source_bytes*1000/len(files)/1_000_000,2),
      'estimated_1000_passage_megabytes_decimal':round(passage_bytes*1000/len(files)/1_000_000,2),
      'model_cost_to_ingest_usd':0,
      'hosting_cost_estimate':'Potentially $0 on a free Neon tier if final database including indexes stays within plan limits; actual database size and compute usage are unmeasured. No valid total-cost figure without a live database.',
      'extrapolation_note':'Linear extrapolation assumes 1000 meetings have the same mean size as this 138-meeting sample and excludes database overhead.'}
    (ROOT/'evaluation/benchmark.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))

if __name__=='__main__':measure()
