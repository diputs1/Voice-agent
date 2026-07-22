export const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export type Citation = {
  source_url: string;
  title: string;
  section: string;
  category: string;
  language?: string | null;
  crawled_at?: string | null;
  valid_until?: string | null;
};

export type ChatDone = {
  thread_id: string;
  answer: string;
  citations: Citation[];
  confidence: number;
  handoff_required: boolean;
  handoff_reason?: string | null;
  recommended_action?: string | null;
};

export async function fetchScribeToken(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/voice/stt-token`, { method: "POST" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = (await response.json()) as { token: string };
  return data.token;
}

export async function fetchVoiceAgentToken(): Promise<string> {
  const response = await fetch(`${API_BASE_URL}/voice/agent-token`, { method: "POST" });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const data = (await response.json()) as { token: string };
  return data.token;
}

export async function playTTS(text: string): Promise<void> {
  const response = await fetch(`${API_BASE_URL}/voice/tts`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const audio = new Audio(url);
  audio.onended = () => URL.revokeObjectURL(url);
  await audio.play();
}
