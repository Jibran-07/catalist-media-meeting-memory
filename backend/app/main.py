"""Catalist meeting memory REST API."""
from pathlib import Path
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from .database import db
from .reasoning import ask, generate_brief, model_configured

app=FastAPI(title='Meeting Memory',version='1.0.0')
allowed=[s.strip() for s in os.getenv('FRONTEND_URL','http://localhost:3000').split(',') if s.strip()]
app.add_middleware(CORSMiddleware,allow_origins=allowed,allow_credentials=False,
    allow_methods=['GET','POST'],allow_headers=['Content-Type'])

ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'data'/'questions.json'
ANSWERS=ROOT/'answers.json'

class AskRequest(BaseModel):
    question: str = Field(min_length=5,max_length=1000)
    team: str | None = None

@app.get('/health')
def health():
    db_status='not_configured'
    if os.getenv('DATABASE_URL'):
        try:
            with db() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT count(*) FROM meetings')
                    db_status=f'connected ({cur.fetchone()[0]} meetings)'
        except Exception:
            db_status='unavailable'
    return {'service':'ok','database':db_status,'reasoning':'configured' if model_configured() else 'evidence_only'}

@app.get('/api/questions')
def get_questions():
    try:
        questions=json.loads(DATA.read_text(encoding='utf-8'))['questions']
        return {'questions':questions}
    except (OSError,KeyError,ValueError):
        raise HTTPException(status_code=503,detail='Assessment questions not loaded')

@app.post('/api/ask')
async def ask_question(req: AskRequest):
    try:
        return await ask(req.question,req.team)
    except Exception as exc:
        # Do not reveal credentials, SQL strings, or connection details.
        raise HTTPException(status_code=503,detail='Meeting search is currently unavailable') from exc

@app.get('/api/brief')
async def get_brief():
    if ANSWERS.exists():
        document=json.loads(ANSWERS.read_text(encoding='utf-8'))
        return {'brief':document['brief'],'source':'validated_submission'}
    try:
        return await generate_brief('TS3012')
    except Exception as exc:
        raise HTTPException(status_code=503,detail='Meeting brief is unavailable') from exc

@app.get('/api/answers')
def get_answers():
    if not ANSWERS.exists():
        raise HTTPException(status_code=404,detail='Assessment answers have not been generated')
    return json.loads(ANSWERS.read_text(encoding='utf-8'))
