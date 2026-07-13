"use client";

import { useState } from "react";
import { Mic, Send, Square } from "lucide-react";
import { ConversationProvider, useConversation } from "@elevenlabs/react";
import { API_BASE_URL, ChatDone, fetchVoiceAgentToken } from "@/lib/api";

type Message = {
  role: "user" | "assistant";
  content: string;
};

export default function Home() {
  return (
    <ConversationProvider>
      <VoiceAgentHome />
    </ConversationProvider>
  );
}

function VoiceAgentHome() {
  const [threadId, setThreadId] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [status, setStatus] = useState("Ready");
  const [voiceError, setVoiceError] = useState<string | null>(null);
  const [liveAgentText, setLiveAgentText] = useState("");
  const [isStarting, setIsStarting] = useState(false);
  const [isSending, setIsSending] = useState(false);

  const conversation = useConversation({
    onConnect: ({ conversationId }) => {
      setVoiceError(null);
      setStatus(`Connected: ${conversationId}`);
      setIsStarting(false);
    },
    onDisconnect: (details) => {
      setStatus(details.reason === "error" ? "Voice error" : "Ready");
      setIsStarting(false);
      setLiveAgentText("");
    },
    onError: (message, context) => showVoiceError("ElevenLabs Agent", context ?? message),
    onMessage: ({ role, message }) => {
      const content = message.trim();
      if (!content) return;
      appendMessage(role === "agent" ? "assistant" : "user", content);
      if (role === "agent") setLiveAgentText("");
    },
    onAgentChatResponsePart: (part) => {
      if (part.type === "start") setLiveAgentText("");
      if (part.type === "delta") setLiveAgentText((current) => current + part.text);
      if (part.type === "stop") setLiveAgentText("");
    },
    onAgentToolRequest: () => setStatus("Retrieving knowledge"),
    onAgentToolResponse: () => setStatus("Agent responding"),
    onModeChange: ({ mode }) => {
      setStatus(mode === "speaking" ? "Agent speaking" : "Listening");
    },
  });

  const isConnected = conversation.status === "connected";

  function appendMessage(role: Message["role"], content: string) {
    setMessages((current) => {
      const last = current[current.length - 1];
      if (last?.role === role && last.content === content) return current;
      return [...current, { role, content }];
    });
  }

  function appendToLastAssistantMessage(text: string) {
    setMessages((current) => {
      const copy = [...current];
      const last = copy[copy.length - 1];
      if (!last || last.role !== "assistant") return current;
      copy[copy.length - 1] = { ...last, content: last.content + text };
      return copy;
    });
  }

  function replaceLastAssistantMessage(text: string) {
    setMessages((current) => {
      const copy = [...current];
      const last = copy[copy.length - 1];
      if (!last || last.role !== "assistant") return current;
      copy[copy.length - 1] = { ...last, content: text };
      return copy;
    });
  }

  function showVoiceError(label: string, error: unknown) {
    setVoiceError(`${label}: ${describeError(error)}`);
    setStatus("Voice error");
    setIsStarting(false);
  }

  async function startVoiceAgent() {
    if (isStarting || conversation.status === "connecting") return;
    try {
      setVoiceError(null);
      setIsStarting(true);
      setStatus("Connecting voice agent");
      await requestMicrophonePermission();
      const conversationToken = await fetchVoiceAgentToken();
      conversation.startSession({
        conversationToken,
        connectionType: "webrtc",
        dynamicVariables: {
          site_id: "default",
        },
      });
    } catch (error) {
      showVoiceError("Cannot start voice agent", error);
      conversation.endSession();
    }
  }

  function stopVoiceAgent() {
    setVoiceError(null);
    setLiveAgentText("");
    conversation.endSession();
    setStatus("Ready");
  }

  async function sendTextMessage() {
    const text = input.trim();
    if (!text || isSending) return;
    setInput("");
    setVoiceError(null);
    setIsSending(true);
    setStatus("LangGraph is thinking");
    setMessages((current) => [...current, { role: "user", content: text }, { role: "assistant", content: "" }]);

    try {
      const response = await fetch(`${API_BASE_URL}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ transcript: text, thread_id: threadId }),
      });

      if (!response.ok || !response.body) {
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
            appendToLastAssistantMessage(String(event.data.text));
          }
          if (event.event === "replace") {
            replaceLastAssistantMessage(String(event.data.text));
          }
          if (event.event === "done") {
            final = event.data as ChatDone;
          }
        }
      }

      if (final) {
        setThreadId(final.thread_id);
        setStatus(final.handoff_required ? "Needs confirmation" : "Ready");
      } else {
        setStatus("Ready");
      }
    } catch (error) {
      replaceLastAssistantMessage(describeError(error));
      setStatus("Error");
    } finally {
      setIsSending(false);
    }
  }

  return (
    <main className="page">
      <header className="topbar">
        <div className="topbar-inner">
          <div className="brand">
            <strong>Vin Agent</strong>
          </div>
          <span className="status">{status}</span>
        </div>
      </header>

      <section className="shell voice-shell">
        <div className="conversation">
          <div className="messages">
            {messages.length === 0 ? (
              <div className="message assistant">
                Xin chào, bạn có thể hỏi về giờ mở cửa, giá vé, show, khu tham quan hoặc dịch vụ tại Vinpearl Safari Phú Quốc.
              </div>
            ) : (
              messages.map((message, index) => (
                <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                  <FormattedMessage text={message.content} />
                </div>
              ))
            )}
            {liveAgentText ? (
              <div className="message assistant live-message">
                <FormattedMessage text={liveAgentText} />
              </div>
            ) : null}
          </div>

          <div className="composer">
            <div className={`live ${voiceError ? "error" : ""}`}>
              {voiceError ?? (isConnected ? "Mic is connected. Send uses text chat." : "Mic is not connected. Send uses text chat.")}
            </div>
            <div className="input-row">
              <textarea
                value={input}
                onChange={(event) => setInput(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" && !event.shiftKey) {
                    event.preventDefault();
                    sendTextMessage();
                  }
                }}
                placeholder="Nhập câu hỏi hoặc sử dụng voice agent..."
              />
              <button
                className={`icon-button ${isConnected ? "active" : ""}`}
                onClick={isConnected ? stopVoiceAgent : startVoiceAgent}
                disabled={isStarting || conversation.status === "connecting"}
                title={isConnected ? "Stop voice agent" : "Start voice agent"}
                type="button"
              >
                {isConnected ? <Square size={18} /> : <Mic size={18} />}
              </button>
              <button
                className="send-button"
                onClick={sendTextMessage}
                disabled={!input.trim() || isSending}
                type="button"
              >
                <Send size={18} />
                Send
              </button>
            </div>
          </div>
        </div>
      </section>
    </main>
  );
}

async function requestMicrophonePermission() {
  if (!navigator.mediaDevices?.getUserMedia) return;
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: {
      echoCancellation: true,
      noiseSuppression: true,
    },
  });
  stream.getTracks().forEach((track) => track.stop());
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

function describeError(error: unknown) {
  if (error instanceof Error) return error.message;
  if (error instanceof Event) return error.type || "browser event";
  if (typeof error === "string") return error;
  if (typeof error === "number") return String(error);
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
