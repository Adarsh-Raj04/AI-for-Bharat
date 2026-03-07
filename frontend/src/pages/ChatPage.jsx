import { useState, useRef, useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import api from "../services/api";
import ChatMessage from "../components/ChatMessage";
import DisclaimerModal from "../components/DisclaimerModal";
import IngestButton from "../components/IngestButton";

export default function ChatPage({
  sessionId,
  onSessionChange,
  onMessagesChange,
  onSessionNameChange,
  onMessagesLoadingChange, // ← notifies App so Sidebar spinner works
}) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const [activeDocFilter, setActiveDocFilter] = useState(null);
  const [messagesLoading, setMessagesLoading] = useState(false); // ← NEW
  const messagesEndRef = useRef(null);
  const { getAccessTokenSilently } = useAuth0();
  const [sessionRefresh, setSessionRefresh] = useState(0);

  useEffect(() => {
    checkDisclaimerStatus();
  }, []);

  useEffect(() => {
    if (sessionId) {
      loadSessionMessages(sessionId);
    } else {
      updateMessages([]);
      setMessagesLoading(false);
    }
  }, [sessionId]);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const pendingNotifyRef = useRef(null);

  const updateMessages = (msgsOrFn) => {
    if (typeof msgsOrFn === "function") {
      setMessages((prev) => {
        const next = msgsOrFn(prev);
        pendingNotifyRef.current = next;
        return next;
      });
    } else {
      setMessages(msgsOrFn);
      pendingNotifyRef.current = msgsOrFn;
    }
  };

  useEffect(() => {
    if (pendingNotifyRef.current !== null) {
      onMessagesChange?.(pendingNotifyRef.current);
      pendingNotifyRef.current = null;
    }
  });

  // ── Load session messages with loading state ─────────────────────────────
  const loadSessionMessages = async (sid) => {
    setMessagesLoading(true);
    onMessagesLoadingChange?.(true); // ← notify App → Layout → Sidebar
    updateMessages([]);
    try {
      const token = await getAccessTokenSilently();
      const response = await api.get(`/sessions/${sid}/messages`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      updateMessages(response.data.messages);
    } catch (error) {
      console.error("Failed to load messages:", error);
      updateMessages([]);
    } finally {
      setMessagesLoading(false);
      onMessagesLoadingChange?.(false); // ← notify App → Layout → Sidebar
    }
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const checkDisclaimerStatus = async () => {
    try {
      const token = await getAccessTokenSilently();
      const response = await api.get("/auth/disclaimer/status", {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (!response.data.accepted) {
        setShowDisclaimer(true);
      }
    } catch (error) {
      console.error("Failed to check disclaimer:", error);
    }
  };

  const handleIngestResult = ({
    summary,
    citation,
    source_id,
    already_existed,
  }) => {
    const alreadyNote = already_existed
      ? "\n\n> *This document was already in the knowledge base — summary generated from existing index.*"
      : "";
    setActiveDocFilter({
      source_id: source_id,
      title: citation?.title || "Uploaded document",
    });
    updateMessages((prev) => [
      ...prev,
      {
        role: "assistant",
        content: summary + alreadyNote,
        citations: citation ? [citation] : [],
        timeline: [],
        confidence: 1.0,
        intent: "summarization",
        streaming: false,
        error: false,
      },
    ]);
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    const withUser = [...messages, userMessage];
    // Stable ID prevents index-shift crash when ingest messages are in the array
    const placeholderId = `streaming-${Date.now()}`;

    updateMessages([
      ...withUser,
      {
        _id: placeholderId,
        role: "assistant",
        content: "",
        citations: [],
        timeline: [],
        confidence: null,
        intent: null,
        streaming: true,
        error: false,
      },
    ]);

    setInput("");
    setLoading(true);

    let accumulatedContent = "";

    const patchPlaceholder = (patch) => {
      updateMessages((prev) => {
        const idx = prev.findIndex((m) => m._id === placeholderId);
        if (idx === -1) return prev;
        const next = [...prev];
        next[idx] = { ...next[idx], ...patch };
        return next;
      });
    };

    setSessionRefresh((prev) => prev + 2);

    try {
      const token = await getAccessTokenSilently();
      const apiUrl = import.meta.env.VITE_API_URL;

      const response = await fetch(`${apiUrl}/api/v1/chat/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          session_id: sessionId,
          message: input,
          stream: true,
          source_filter: activeDocFilter?.source_id ?? null,
        }),
      });

      if (!response.ok) throw new Error(`HTTP ${response.status}`);

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const frames = buffer.split("\n\n");
        buffer = frames.pop();

        for (const frame of frames) {
          if (!frame.trim()) continue;

          let eventType = null;
          let dataStr = null;

          for (const line of frame.split("\n")) {
            if (line.startsWith("event: ")) eventType = line.slice(7).trim();
            if (line.startsWith("data: ")) dataStr = line.slice(6).trim();
          }

          if (!eventType || !dataStr) continue;

          let payload;
          try {
            payload = JSON.parse(dataStr);
          } catch {
            continue;
          }

          switch (eventType) {
            case "session":
              if (onSessionChange && payload.session_id)
                onSessionChange(payload.session_id);
              break;
            case "citations":
              patchPlaceholder({
                citations: payload.citations ?? [],
                timeline: payload.timeline ?? [],
              });
              break;
            case "text":
              accumulatedContent += payload.text ?? "";
              patchPlaceholder({ content: accumulatedContent });
              break;
            case "done":
              patchPlaceholder({
                id: payload.message_id,
                confidence: payload.confidence,
                intent: payload.intent,
                streaming: false,
              });
              if (onSessionChange && payload.session_id)
                onSessionChange(payload.session_id);
              break;

            case "rename":
              // Backend generated a smart name — update Sidebar live
              if (payload.session_name)
                onSessionNameChange?.(payload.session_name);
              break;
            case "error":
              patchPlaceholder({
                content: payload.message ?? "An error occurred.",
                error: true,
                streaming: false,
              });
              break;
            default:
              break;
          }
        }
      }
    } catch (error) {
      console.error("Streaming failed:", error);
      let errorContent =
        "Sorry, I encountered an error processing your request. Please try again.";
      if (error?.message === "HTTP 400") {
        errorContent =
          "Your query appears to contain personal health information (PHI). " +
          "Please remove any patient identifiers, phone numbers, or personal details " +
          "and rephrase as a general research question.";
      }
      patchPlaceholder({
        content: errorContent,
        error: true,
        streaming: false,
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6">
          <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
            {/* ── Loading spinner while fetching session messages ── */}
            {messagesLoading ? (
              <div className="flex flex-col items-center justify-center py-32 gap-3">
                <div className="flex items-end gap-1 h-7">
                  {[0, 1, 2, 3, 4].map((i) => (
                    <div
                      key={i}
                      className="w-1.5 rounded-full bg-primary-400 dark:bg-primary-500"
                      style={{
                        height: "100%",
                        animation: "chatLoadBar 0.9s ease-in-out infinite",
                        animationDelay: `${i * 0.13}s`,
                      }}
                    />
                  ))}
                </div>
                <p className="text-sm text-gray-400 dark:text-gray-500">
                  Loading messages…
                </p>
                <style>{`
                  @keyframes chatLoadBar {
                    0%, 100% { transform: scaleY(0.25); opacity: 0.35; }
                    50%      { transform: scaleY(1);    opacity: 1; }
                  }
                `}</style>
              </div>
            ) : messages.length === 0 ? (
              <div className="text-center py-8 sm:py-12 px-4">
                <svg
                  className="w-12 h-12 sm:w-16 sm:h-16 mx-auto text-gray-400 dark:text-gray-600 mb-3 sm:mb-4"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M8 10h.01M12 10h.01M16 10h.01M9 16H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-5l-5 5v-5z"
                  />
                </svg>
                <h2 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-gray-100 mb-2">
                  Welcome to MedResearch AI
                </h2>
                <p className="text-sm sm:text-base text-gray-600 dark:text-gray-400 mb-4 sm:mb-6">
                  Your intelligent research assistant for life sciences
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 sm:gap-4 max-w-5xl mx-auto">
                  <ExampleQuery
                    icon="🩺"
                    text="Crohn's disease treatment options"
                    onClick={() =>
                      setInput(
                        "What are the latest treatment options for Crohn's disease?",
                      )
                    }
                  />
                  <ExampleQuery
                    icon="💊"
                    text="Rheumatoid arthritis therapy"
                    onClick={() =>
                      setInput(
                        "What are the current therapies for rheumatoid arthritis?",
                      )
                    }
                  />
                  <ExampleQuery
                    icon="🧬"
                    text="Type 2 diabetes management"
                    onClick={() =>
                      setInput(
                        "What are the best practices for managing type 2 diabetes?",
                      )
                    }
                  />
                  <ExampleQuery
                    icon="🫀"
                    text="Chronic kidney disease treatment"
                    onClick={() =>
                      setInput(
                        "What are the treatment approaches for chronic kidney disease?",
                      )
                    }
                  />
                  <ExampleQuery
                    icon="🧠"
                    text="Major depressive disorder therapy"
                    onClick={() =>
                      setInput(
                        "What are the therapeutic options for major depressive disorder?",
                      )
                    }
                  />
                </div>
              </div>
            ) : (
              messages
                .filter(Boolean)
                .map((message, index) => (
                  <ChatMessage key={message._id || index} message={message} />
                ))
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 sm:px-4 py-3 sm:py-4 safe-area-bottom transition-colors">
          <div className="max-w-4xl mx-auto">
            {/* Document filter banner */}
            {activeDocFilter && (
              <div className="flex items-center justify-between gap-2 mb-2 px-3 py-2 rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-violet-500 dark:text-violet-400 text-sm flex-shrink-0">
                    📄
                  </span>
                  <span className="text-xs text-violet-700 dark:text-violet-300 truncate">
                    Chatting with:{" "}
                    <span className="font-medium">
                      {activeDocFilter.title.length > 60
                        ? activeDocFilter.title.slice(0, 60) + "…"
                        : activeDocFilter.title}
                    </span>
                  </span>
                </div>
                <button
                  type="button"
                  onClick={() => setActiveDocFilter(null)}
                  className="flex-shrink-0 text-xs text-violet-500 dark:text-violet-400 hover:text-violet-700 dark:hover:text-violet-200 font-medium whitespace-nowrap"
                >
                  ✕ Clear
                </button>
              </div>
            )}

            <form
              onSubmit={handleSendMessage}
              className="flex flex-col sm:flex-row gap-2 sm:gap-4"
            >
              <div className="flex items-center gap-2 flex-1">
                <IngestButton
                  sessionId={sessionId}
                  onSummaryReady={handleIngestResult}
                  disabled={loading}
                />
                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about research, or use 📎 to summarize a document..."
                  className="flex-1 px-3 sm:px-4 py-2.5 sm:py-3 text-sm sm:text-base border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-primary-400 focus:border-transparent transition-colors"
                  disabled={loading}
                />
              </div>
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-4 sm:px-6 py-2.5 sm:py-3 bg-primary-600 dark:bg-primary-700 text-white rounded-lg hover:bg-primary-700 dark:hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium text-sm sm:text-base active:scale-95 flex items-center justify-center gap-2"
              >
                <span>Send</span>
                <svg
                  className="w-4 h-4 rotate-90"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                  />
                </svg>
              </button>
            </form>

            <div className="flex flex-col sm:flex-row items-center justify-center gap-2 sm:gap-4 mt-3 text-xs text-gray-500">
              <p className="text-center px-2">
                ⚠️ This tool is for research purposes only and does not provide
                medical advice. Always verify information with trusted sources.
              </p>
              <span className="hidden sm:inline text-gray-300">•</span>
              <p className="text-center">
                Built with <span className="text-red-500">❤️</span> by{" "}
                <a
                  href="https://www.linkedin.com/in/Adarsh-Raj04"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-primary-600 hover:text-primary-700 hover:underline"
                >
                  Adarsh Raj
                </a>
              </p>
            </div>
          </div>
        </div>
      </div>

      <DisclaimerModal
        isOpen={showDisclaimer}
        onClose={() => setShowDisclaimer(false)}
      />
    </>
  );
}

function ExampleQuery({ icon, text, onClick }) {
  return (
    <button
      onClick={onClick}
      className="p-3 sm:p-4 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:border-primary-500 dark:hover:border-primary-400 hover:shadow-md transition-all text-left active:scale-95"
    >
      <div className="text-xl sm:text-2xl mb-1 sm:mb-2">{icon}</div>
      <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">
        {text}
      </p>
    </button>
  );
}
