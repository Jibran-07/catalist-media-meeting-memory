"use client";
import {useEffect, useState} from "react";
import {api, type Answer, type Question} from "../lib/api";

type Section = "ask" | "assessment" | "brief";
const sampleQuestion = "What decisions changed in Team TS3012 between the first and third meetings?";

function Sources({result}: {result: Answer}) {
  return <div className="mt-6">
    {result.citations?.length > 0 && <>
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-600 mb-3">Sources</h3>
      <div className="space-y-3">{result.citations.map((cite, i) =>
        <div key={`${cite.meeting_id}-${i}`} className="border border-border rounded-md p-4 bg-white">
          <p className="text-sm font-semibold mb-2">Meeting {cite.meeting_id}</p>
          <blockquote className="text-sm text-slate-700 leading-6">“{cite.quote}”</blockquote>
        </div>)}</div>
    </>}
    {result.status === "evidence_only" && <>
      <p className="mt-3 text-sm text-amber-900">The search is working, but an inference model is not configured. These are source excerpts, not a generated answer.</p>
      <h3 className="text-sm font-semibold uppercase tracking-wide text-slate-600 mt-5 mb-3">Matching excerpts</h3>
      <div className="space-y-3 max-h-[440px] overflow-y-auto">{result.evidence?.slice(0, 12).map((item, i) =>
        <div className="border border-border rounded-md bg-white p-3 text-sm" key={i}>
          <p className="font-semibold text-slate-600">{item.meeting_id} · {item.timecode} · Speaker {item.speaker}</p>
          <p className="mt-2 leading-6">{item.text}</p>
        </div>)}</div>
    </>}
    {result.status === "citation_error" && <p className="text-red-700 text-sm mt-4">The draft answer was withheld because its supporting quotations could not be verified.</p>}
  </div>;
}

export default function Home() {
  const [section, setSection] = useState<Section>("ask");
  const [question, setQuestion] = useState(sampleQuestion);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [questions, setQuestions] = useState<Question[]>([]);
  const [savedAnswers, setSavedAnswers] = useState<(Omit<Answer, "status"> & {question_id: string})[]>([]);
  const [chosen, setChosen] = useState("Q01");
  const [brief, setBrief] = useState("");
  const [briefLoading, setBriefLoading] = useState(false);

  useEffect(() => {
    api.questions().then(d => setQuestions(d.questions)).catch(() => {});
    api.savedAnswers().then(d => setSavedAnswers(d.answers)).catch(() => {});
  }, []);

  async function submit(q: string, team?: string) {
    setLoading(true);setError("");setAnswer(null);
    try { setAnswer(await api.ask(q, team)); }
    catch(e) { setError(e instanceof Error ? e.message : "Search failed. Please try again."); }
    finally { setLoading(false); }
  }
  async function showBrief() {
    setBriefLoading(true);setError("");
    try { const data = await api.brief(); setBrief(data.brief ?? data.answer ?? "Brief not available."); }
    catch(e) { setError(e instanceof Error ? e.message : "Could not load the brief."); }
    finally { setBriefLoading(false); }
  }
  function changeTab(tab: Section) {
    setSection(tab); setError(""); setAnswer(null);
    if (tab === "brief") void showBrief();
  }
  const selected = questions.find(q => q.question_id === chosen);
  return <main className="max-w-4xl mx-auto px-5 sm:px-8 py-9 sm:py-12">
    <header className="mb-8">
      <h1 className="text-3xl font-semibold tracking-tight">Meeting Memory</h1>
      <p className="text-slate-600 mt-2 text-sm">Search meeting transcripts and review decisions with source references.</p>
    </header>
    <nav aria-label="Main sections" className="flex gap-2 border-b border-border mb-7 overflow-x-auto">
      {([['ask','Ask a question'],['assessment','Assessment questions'],['brief','Meeting brief']] as [Section,string][]).map(([tab,label]) =>
        <button type="button" key={tab} onClick={() => changeTab(tab)}
          className={`px-4 py-3 text-sm whitespace-nowrap border-b-2 ${section === tab ? 'border-slate-800 font-semibold' : 'border-transparent text-slate-600 hover:text-slate-900'}`}>
          {label}</button>)}
    </nav>
    {section === "ask" && <section>
      <h2 className="text-lg font-semibold mb-4">Ask a question</h2>
      <form onSubmit={e => {e.preventDefault(); void submit(question);}}>
        <label htmlFor="query" className="block text-sm font-medium mb-2">What would you like to know?</label>
        <textarea id="query" className="w-full min-h-32 p-3 border border-border rounded-md bg-white focus:outline focus:outline-2 focus:outline-slate-500"
          value={question} onChange={e => setQuestion(e.target.value)} placeholder="Ask about a decision, commitment, or budget" required minLength={5} maxLength={1000}/>
        <button type="submit" disabled={loading || question.trim().length < 5}
          className="bg-slate-800 text-white rounded-md px-5 py-2.5 text-sm font-medium mt-3 hover:bg-slate-700">{loading ? "Searching..." : "Search meetings"}</button>
      </form>
    </section>}
    {section === "assessment" && <section>
      <h2 className="text-lg font-semibold mb-2">Assessment questions</h2>
      <p className="text-sm text-slate-600 mb-5">Review source-checked submission answers or search the database again when it is connected.</p>
      <label htmlFor="questions" className="block mb-2 text-sm font-medium">Select question</label>
      <select id="questions" value={chosen} onChange={e => {setChosen(e.target.value);setAnswer(null);}}
        className="border border-border rounded-md p-3 bg-white w-full mb-4">
        {questions.length ? questions.map(q => <option key={q.question_id} value={q.question_id}>{q.question_id} · {q.team}</option>) : <option>Questions unavailable</option>}
      </select>
      {selected && <div className="border border-border bg-white rounded-md p-4 mb-4"><p className="text-sm leading-7">{selected.question}</p></div>}
      <div className="flex flex-wrap items-center gap-3">
        <button type="button" disabled={!savedAnswers.some(a => a.question_id === chosen)} onClick={() => {
          const saved=savedAnswers.find(a => a.question_id === chosen);
          if (saved) {setAnswer({...saved,status:"answered"});setError("");}
        }} className="bg-slate-800 text-white rounded-md px-5 py-2.5 text-sm font-medium hover:bg-slate-700 disabled:opacity-50">Show reviewed answer</button>
        <button type="button" disabled={loading || !selected} onClick={() => selected && void submit(selected.question,selected.team)}
          className="border border-border bg-white rounded-md px-5 py-2.5 text-sm font-medium hover:bg-slate-50">{loading ? "Searching..." : "Run new search"}</button>
      </div>
    </section>}
    {section === "brief" && <section>
      <h2 className="text-lg font-semibold mb-2">Team TS3012 meeting brief</h2>
      <p className="text-sm text-slate-600 mb-5">Prepared from meetings a, b, and c, before the final meeting.</p>
      {briefLoading ? <p>Loading meeting brief...</p> : <article className="bg-white border border-border rounded-md px-5 py-4 whitespace-pre-line text-sm leading-7">{brief || 'Brief not available.'}</article>}
      <button type="button" onClick={() => void showBrief()} disabled={briefLoading}
        className="mt-4 border border-border px-4 py-2 rounded-md text-sm bg-white">Reload brief</button>
    </section>}
    {error && <p role="alert" className="text-red-800 bg-red-50 border border-red-200 mt-6 p-3 rounded-md text-sm">{error}</p>}
    {answer && <section className="mt-8 border-t border-border pt-6" aria-live="polite">
      <h2 className="text-lg font-semibold mb-3">Answer</h2>
      <p className="text-sm leading-7">{answer.answer}</p>
      <Sources result={answer}/>
    </section>}
    <footer className="text-xs text-slate-500 mt-16 border-t border-border pt-5">Source: AMI Meeting Corpus, CC BY 4.0.</footer>
  </main>;
}
