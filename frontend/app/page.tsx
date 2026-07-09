"use client";

import { useMemo, useState } from "react";
import { ExternalLink, Mic, Send, Square, Volume2 } from "lucide-react";
import { useScribe } from "@elevenlabs/react";
import { API_BASE_URL, Citation, ChatDone, fetchScribeToken, playTTS } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [sources, setSources] = useState<Citation[]>([]);
  const [handoff, setHandoff] = useState<Pick<ChatDone, "handoff_reason" | "recommended_action"> | null>(null);
  const [status, setStatus] = useState("Ready");
  const [isSending, setIsSending] = useState(false);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  function showVoiceError(label: string, error: unknown) {
    setVoiceError(`${label}: ${describeError(error)}`);
    setStatus("Voice error");
  }

  const scribe = useScribe({
    modelId: "scribe_v2_realtime",
    onConnect: () => {
      setVoiceError(null);
      setStatus("Listening");
    },
    onDisconnect: () => {
      setStatus((current) => (current === "Voice error" ? current : "Ready"));
    },
    onError: (error) => showVoiceError("Voice websocket", error),
    onAuthError: (data) => showVoiceError("ElevenLabs auth", data),
    onQuotaExceededError: (data) => showVoiceError("ElevenLabs quota exceeded", data),
    onRateLimitedError: (data) => showVoiceError("ElevenLabs rate limited", data),
    onTranscriberError: (data) => showVoiceError("ElevenLabs transcriber", data),
    onInputError: (data) => showVoiceError("Microphone input", data),
    onUnacceptedTermsError: (data) => showVoiceError("ElevenLabs terms", data),
    onResourceExhaustedError: (data) => showVoiceError("ElevenLabs resource exhausted", data),
    onSessionTimeLimitExceededError: (data) => showVoiceError("ElevenLabs session timeout", data),
    onChunkSizeExceededError: (data) => showVoiceError("Audio chunk too large", data),
    onInsufficientAudioActivityError: (data) => showVoiceError("No speech detected", data),
    onQueueOverflowError: (data) => showVoiceError("Audio queue overflow", data),
    onCommitThrottledError: (data) => showVoiceError("Speech commit throttled", data),
    onPartialTranscript: (data) => {
      setVoiceError(null);
      setInput(data.text);
    },
    onCommittedTranscript: (data) => {
      setVoiceError(null);
      setInput(data.text);
      void sendQuestion(data.text);
    },
  });

  const lastAnswer = useMemo(
    () => [...messages].reverse().find((message) => message.role === "assistant")?.content ?? "",
    [messages],
  );

  async function startVoice() {
    try {
      setVoiceError(null);
      setStatus("Connecting microphone");
      const token = await fetchScribeToken();
      await scribe.connect({
        token,
        microphone: {
          echoCancellation: true,
          noiseSuppression: true,
        },
        includeLanguageDetection: true,
        minSpeechDurationMs: 250,
        minSilenceDurationMs: 700,
        vadSilenceThresholdSecs: 1.2,
      });
    } catch (error) {
      showVoiceError("Cannot start voice", error);
      scribe.disconnect();
    }
  }

  async function stopVoice() {
    setVoiceError(null);
    scribe.disconnect();
    setStatus("Ready");
  }

  async function sendQuestion(question = input.trim()) {
    if (!question || isSending) return;

    setInput("");
    setIsSending(true);
    setStatus("LangGraph is thinking");
    setMessages((current) => [...current, { role: "user", content: question }, { role: "assistant", content: "" }]);

    const response = await fetch(`${API_BASE_URL}/chat/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ transcript: question, thread_id: threadId }),
    });

    if (!response.ok || !response.body) {
      setIsSending(false);
      setStatus("Error");
      throw new Error(await response.text());
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let final: ChatDone | null = null;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const events = buffer.split("\n\n");
      buffer = events.pop() ?? "";

      for (const eventText of events) {
        const event = parseSse(eventText);
        if (!event) continue;
        if (event.event === "status") {
          setStatus(String(event.data.node));
        }
        if (event.event === "token") {
          setMessages((current) => {
            const copy = [...current];
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = { ...last, content: last.content + String(event.data.text) };
            return copy;
          });
        }
        if (event.event === "replace") {
          setMessages((current) => {
            const copy = [...current];
            const last = copy[copy.length - 1];
            copy[copy.length - 1] = { ...last, content: String(event.data.text) };
            return copy;
          });
        }
        if (event.event === "done") {
          final = event.data as ChatDone;
        }
      }
    }

    if (final) {
      setThreadId(final.thread_id);
      setSources(final.citations);
      setHandoff(
        final.handoff_required
          ? {
              handoff_reason: final.handoff_reason,
              recommended_action: final.recommended_action,
            }
          : null,
      );
      setStatus(final.handoff_required ? "Needs confirmation" : "Ready");
    } else {
      setStatus("Ready");
    }
    setIsSending(false);
  }

  async function speakLastAnswer() {
    if (!lastAnswer) return;
    setStatus("Speaking");
    await playTTS(lastAnswer);
    setStatus("Ready");
  }

  return (
    <main className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <strong>Vin Agent</strong>
            <span>LangGraph voice Q&A for Vinpearl Safari Phu Quoc</span>
          </div>
          <span className="status">{status}</span>
        </div>
      </header>

      <section className="shell">
        <div className="conversation">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="message assistant">
                Xin chào, bạn có thể hỏi về giờ mở cửa, trải nghiệm Safari, show, khu tham quan hoặc dịch vụ trong công viên.
              </div>
            ) : (
              messages.map((message, index) => (
                <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  {message.content ? <FormattedMessage text={message.content} /> : "..."}
                </div>
              ))
            )}
          </div>

          <div className="composer">
            <div className={`live ${voiceError ? "error" : ""}`}>
              {voiceError ?? (scribe.partialTranscript ? `Live: ${scribe.partialTranscript}` : " ")}
            </div>
            <div className="input-row">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                placeholder="Nhập hoặc nói câu hỏi của bạn"
              />
              <button
                className={`icon-button ${scribe.isConnected ? "active" : ""}`}
                onClick={scribe.isConnected ? stopVoice : startVoice}
                disabled={scribe.status === "connecting"}
                title={scribe.isConnected ? "Stop recording" : "Start recording"}
                type="button"
              >
                {scribe.isConnected ? <Square size={18} /> : <Mic size={18} />}
              </button>
              <button className="send-button" onClick={() => sendQuestion()} disabled={isSending} type="button">
                <Send size={18} />
                Send
              </button>
            </div>
            <button className="icon-button" onClick={speakLastAnswer} disabled={!lastAnswer} type="button">
              <Volume2 size={18} />
              Speak last answer
            </button>
          </div>
        </div>

        <aside className="side">
          <h2>Sources</h2>
          {handoff ? <HandoffNotice handoff={handoff} /> : null}
          <div className="source-list">
            {sources.length === 0 ? (
              <span className="status">Sources appear after the first answer.</span>
            ) : (
              sources.map((source, index) => (
                <div className="source" key={`${source.source_url}-${index}`}>
                  <strong>{sourceLabel(source)}</strong>
                  <div className="source-meta">
                    <span>{source.category}</span>
                    {source.language ? <span>{source.language.toUpperCase()}</span> : null}
                  </div>
                  <a href={source.source_url} target="_blank" rel="noreferrer">
                    {readableUrl(source.source_url)}
                  </a>
                </div>
              ))
            )}
          </div>
        </aside>
      </section>
    </main>
  );
}

function HandoffNotice({ handoff }: { handoff: Pick<ChatDone, "handoff_reason" | "recommended_action"> }) {
  const href =
    handoff.recommended_action === "contact_hotline_or_booking"
      ? "https://booking.vinwonders.com/"
      : "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/";

  return (
    <div className="handoff">
      <strong>{handoffTitle(handoff.handoff_reason)}</strong>
      <span>Thông tin này cần xác nhận lại trước khi đặt dịch vụ.</span>
      <a href={href} target="_blank" rel="noreferrer">
        <ExternalLink size={15} />
        {handoff.recommended_action === "contact_hotline_or_booking" ? "Mở trang booking" : "Mở nguồn chính thức"}
      </a>
    </div>
  );
}

function handoffTitle(reason?: string | null) {
  if (reason === "ungrounded_answer") return "Answer needs verification";
  if (reason === "low_confidence") return "Low confidence";
  return "Freshness check required";
}

function FormattedMessage({ text }: { text: string }) {
  return (
    <>
      {text.split("\n").map((line, index) => (
        <p key={`${line}-${index}`}>{formatInline(line)}</p>
      ))}
    </>
  );
}

function formatInline(text: string) {
  const parts = text.split(/(\*\*[^*]+\*\*)/g);
  return parts.map((part, index) => {
    if (part.startsWith("**") && part.endsWith("**")) {
      return <strong key={`${part}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${part}-${index}`}>{part}</span>;
  });
}

function sourceLabel(source: Citation) {
  const section = source.section?.trim();
  const title = source.title?.trim();
  if (section && isReadableLabel(section)) return cleanLabel(section);
  if (title) return cleanLabel(title);
  return readableUrl(source.source_url);
}

function cleanLabel(value: string) {
  return value
    .split("|")
    .map((part) => part.trim())
    .filter(Boolean)
    .join(" - ");
}

function isReadableLabel(value: string) {
  const lowered = value.toLowerCase();
  return (
    value.length >= 12 &&
    value[0] !== value[0].toLowerCase() &&
    !value.includes("=") &&
    !value.includes("&") &&
    !lowered.includes("http") &&
    !lowered.includes("utm_") &&
    !lowered.includes("redirecturi") &&
    !lowered.includes("copy to clipboard") &&
    !lowered.includes("đăng nhập") &&
    !lowered.includes("đăng ký") &&
    !lowered.includes("log in") &&
    !lowered.includes("register")
  );
}

function readableUrl(value: string) {
  try {
    const url = new URL(value);
    return `${url.hostname}${url.pathname}`;
  } catch {
    return value;
  }
}

function describeError(error: unknown) {
  if (error instanceof Error) return error.message;
  if (error instanceof Event) return error.type || "browser event";
  if (typeof error === "string") return error;
  if (error && typeof error === "object") {
    const maybeError = error as { error?: unknown; message?: unknown; code?: unknown; reason?: unknown };
    const parts = [maybeError.error, maybeError.message, maybeError.code, maybeError.reason]
      .map((part) => (typeof part === "string" || typeof part === "number" ? String(part) : ""))
      .filter(Boolean);
    if (parts.length > 0) return parts.join(" - ");
    return JSON.stringify(error);
  }
  return "Unknown error";
}

function parseSse(eventText: string): { event: string; data: Record<string, unknown> } | null {
  const eventLine = eventText.split("\n").find((line) => line.startsWith("event: "));
  const dataLine = eventText.split("\n").find((line) => line.startsWith("data: "));
  if (!eventLine || !dataLine) return null;
  return {
    event: eventLine.replace("event: ", ""),
    data: JSON.parse(dataLine.replace("data: ", "")),
  };
}
