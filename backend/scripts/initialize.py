"""python -m scripts.initialize (from backend directory)."""
import argparse
import json
from pathlib import Path
from app.ingestion import ingest

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--transcripts',type=Path,default=Path(__file__).resolve().parents[2]/'data'/'transcripts')
    args=ap.parse_args()
    print(json.dumps(ingest(args.transcripts), indent=2))
