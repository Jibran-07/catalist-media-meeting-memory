export type Citation = { meeting_id: string; quote: string };
export type SourceTurn = { meeting_id: string; turn_index: number; timecode: string; speaker: string; text: string };
export type Answer = { answer: string; citations: Citation[]; status: string; evidence?: SourceTurn[]; details?: string[] };
export type Question = { question_id: string; team: string; question: string };
const base = (process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000").replace(/\/$/, "");
async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(base + path, {...options, cache: "no-store"});
  if (!response.ok) {
    const data = await response.json().catch(() => null);
    throw new Error(data?.detail ?? `Server error (${response.status})`);
  }
  return response.json() as Promise<T>;
}
export const api = {
  ask: (question: string, team?: string) => request<Answer>("/api/ask", {method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify({question, team})}),
  questions: () => request<{questions: Question[]}>("/api/questions"),
  brief: () => request<{brief?:string; answer?:string; source?:string; status?:string}>("/api/brief"),
  savedAnswers: () => request<{answers: (Omit<Answer, "status"> & {question_id: string})[]}>("/api/answers")
};
