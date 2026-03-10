import { useState, useRef, useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { useParams, useNavigate } from "react-router-dom";
import api from "../services/api";
import ChatMessage from "../components/ChatMessage";
import DisclaimerModal from "../components/DisclaimerModal";
import IngestButton from "../components/IngestButton";
import ArtifactsMenu from "../components/ArtifactsMenu";
import {
  getRecentDocs,
  upsertRecentDoc,
  removeRecentDoc,
} from "../services/recentDocs";

export default function ChatPage({
  onMessagesChange,
  onSessionNameChange,
  onMessagesLoadingChange,
}) {
  const { sessionId } = useParams();
  const navigate = useNavigate();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(false);

  // ── Doc filter state ─────────────────────────────────────────────────────
  const [activeDocFilter, setActiveDocFilter] = useState(null);
  const [compareFilters, setCompareFilters] = useState(null);
  const [recentDocs, setRecentDocs] = useState(() => getRecentDocs());

  const [messagesLoading, setMessagesLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const { getAccessTokenSilently } = useAuth0();
  const isStreamingRef = useRef(false);
  const sessionAssignedRef = useRef(false);

  // ── Navigate to new session URL ───────────────────────────────────────────
  const handleSessionChange = (newSessionId) => {
    if (newSessionId && newSessionId !== sessionId) {
      sessionAssignedRef.current = true;
      navigate(`/chat/${newSessionId}`, { replace: !sessionId });
    }
  };

  useEffect(() => {
    checkDisclaimerStatus();
  }, []);

  useEffect(() => {
    if (sessionId) {
      if (sessionAssignedRef.current) {
        sessionAssignedRef.current = false;
        return;
      }
      if (!isStreamingRef.current) {
        setActiveDocFilter(null);
        setCompareFilters(null);
        loadSessionMessages(sessionId);
      }
    } else {
      updateMessages([]);
      setMessagesLoading(false);
      setActiveDocFilter(null);
      setCompareFilters(null);
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

  const loadSessionMessages = async (sid) => {
    setMessagesLoading(true);
    onMessagesLoadingChange?.(true);
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
      onMessagesLoadingChange?.(false);
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
      if (!response.data.accepted) setShowDisclaimer(true);
    } catch (error) {
      console.error("Failed to check disclaimer:", error);
    }
  };

  // ── Ingest result handler ─────────────────────────────────────────────────
  const handleIngestResult = ({
    summary,
    citation,
    source_id,
    already_existed,
    session_id,
    session_name,
  }) => {
    const alreadyNote = already_existed
      ? "\n\n> *This research paper was already in the knowledge base.*"
      : "";
    const docEntry = {
      source_id,
      title: citation?.title || "Uploaded research paper",
    };

    setActiveDocFilter(docEntry);
    setCompareFilters(null);

    upsertRecentDoc(docEntry);
    setRecentDocs(getRecentDocs());

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

    if (session_id) handleSessionChange(session_id);
    if (session_name) onSessionNameChange?.(session_name);
  };

  // ── ArtifactsMenu handler ─────────────────────────────────────────────────
  const handleArtifactSelect = (single, compare) => {
    setActiveDocFilter(single);
    setCompareFilters(compare);
  };

  const handleRemoveDoc = (source_id) => {
    removeRecentDoc(source_id);
    setRecentDocs(getRecentDocs());
    if (activeDocFilter?.source_id === source_id) setActiveDocFilter(null);
    if (compareFilters?.some((d) => d.source_id === source_id))
      setCompareFilters(null);
  };

  // ── Send message ──────────────────────────────────────────────────────────
  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const placeholderId = `streaming-${Date.now()}`;
    const compareDocs = compareFilters ? [...compareFilters] : null;

    updateMessages((prev) => [
      ...prev,
      { role: "user", content: input, compare_docs: compareDocs },
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

    const sentInput = input;
    setInput("");
    setLoading(true);
    isStreamingRef.current = true;

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

    try {
      const token = await getAccessTokenSilently();
      const apiUrl = import.meta.env.VITE_API_URL;

      const body = {
        session_id: sessionId,
        message: sentInput,
        stream: true,
        source_filter: compareFilters
          ? null
          : (activeDocFilter?.source_id ?? null),
        compare_filters: compareFilters
          ? compareFilters.map((d) => d.source_id)
          : null,
        compare_titles: compareFilters
          ? compareFilters.map((d) => d.title)
          : null,
      };

      const response = await fetch(`${apiUrl}/api/v1/chat/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify(body),
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
          let eventType = null,
            dataStr = null;
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
              if (payload.session_id) handleSessionChange(payload.session_id);
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
              if (payload.session_id) handleSessionChange(payload.session_id);
              break;
            case "rename":
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
          "Your query appears to contain personal health information (PHI). Please remove any patient identifiers and rephrase as a general research question.";
      }
      patchPlaceholder({
        content: errorContent,
        error: true,
        streaming: false,
      });
    } finally {
      setLoading(false);
      isStreamingRef.current = false;
    }
  };

  // ── Active filter status bar label ────────────────────────────────────────
  const filterBar = () => {
    if (compareFilters?.length === 2) {
      return (
        <div className="flex items-center justify-between gap-2 mb-2 px-3 py-2 rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-violet-500 text-sm flex-shrink-0">⚖️</span>
            <span className="text-xs text-violet-700 dark:text-violet-300 truncate">
              Comparing:{" "}
              <span className="font-medium">
                {compareFilters[0].title.length > 28
                  ? compareFilters[0].title.slice(0, 28) + "…"
                  : compareFilters[0].title}
              </span>
              <span className="mx-1 opacity-60">vs</span>
              <span className="font-medium">
                {compareFilters[1].title.length > 28
                  ? compareFilters[1].title.slice(0, 28) + "…"
                  : compareFilters[1].title}
              </span>
            </span>
          </div>
          <button
            type="button"
            onClick={() => {
              setCompareFilters(null);
              setActiveDocFilter(null);
            }}
            className="flex-shrink-0 text-xs text-violet-500 dark:text-violet-400 hover:text-violet-700 dark:hover:text-violet-200 font-medium"
          >
            ✕ Clear
          </button>
        </div>
      );
    }
    if (activeDocFilter) {
      return (
        <div className="flex items-center justify-between gap-2 mb-2 px-3 py-2 rounded-lg bg-violet-50 dark:bg-violet-900/20 border border-violet-200 dark:border-violet-700">
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-violet-500 text-sm flex-shrink-0">📄</span>
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
            className="flex-shrink-0 text-xs text-violet-500 dark:text-violet-400 hover:text-violet-700 dark:hover:text-violet-200 font-medium"
          >
            ✕ Clear
          </button>
        </div>
      );
    }
    return null;
  };

  const inputPlaceholder =
    compareFilters?.length === 2
      ? "Ask a question to compare both Research Papers…"
      : activeDocFilter
        ? "Ask about this document…"
        : "Upload a research paper or ask a question...";

  return (
    <>
      <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6">
          <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
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
                <style>{`@keyframes chatLoadBar { 0%,100%{transform:scaleY(.25);opacity:.35} 50%{transform:scaleY(1);opacity:1} }`}</style>
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

        {/* ── Input bar ── */}
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 sm:px-4 py-3 sm:py-4 safe-area-bottom transition-colors">
          <div className="max-w-4xl mx-auto space-y-2">
            {filterBar()}

            <form
              onSubmit={handleSendMessage}
              className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2"
            >
              <div className="w-full sm:w-auto sm:flex-shrink-0">
                <ArtifactsMenu
                  docs={recentDocs}
                  activeFilter={activeDocFilter}
                  compareFilters={compareFilters}
                  onSelect={handleArtifactSelect}
                  onRemoveDoc={handleRemoveDoc}
                />
              </div>

              <div className="flex flex-1 items-center gap-1 rounded-full border border-gray-300 dark:border-gray-600 bg-gray-50 dark:bg-gray-700 pl-1 pr-2 py-1.5 focus-within:ring-2 focus-within:ring-primary-500 focus-within:border-transparent transition-all">
                <IngestButton
                  sessionId={sessionId}
                  onSummaryReady={handleIngestResult}
                  disabled={loading}
                />

                <input
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder={inputPlaceholder}
                  className="flex-1 min-w-0 bg-transparent text-sm text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none"
                  disabled={loading}
                  autoFocus
                />

                <button
                  type="submit"
                  disabled={loading || !input.trim()}
                  className="flex-shrink-0 p-1.5 rounded-full transition-all disabled:opacity-30 disabled:cursor-not-allowed"
                >
                  {compareFilters?.length === 2 ? (
                    <span className="flex items-center gap-1 px-2 py-0.5 text-xs font-semibold text-violet-600 dark:text-violet-400">
                      ⚖️ Compare
                    </span>
                  ) : (
                    <svg
                      viewBox="0 0 24 24"
                      className={`w-5 h-5 transition-colors ${input.trim() && !loading ? "text-primary-600 dark:text-primary-400" : "text-gray-400 dark:text-gray-600"}`}
                    >
                      <polygon points="5,3 19,12 5,21" fill="currentColor" />
                    </svg>
                  )}
                </button>
              </div>
            </form>

            {/* Footer */}
            <div className="flex flex-col sm:flex-row items-center justify-center gap-1 sm:gap-4 pt-1 text-xs text-gray-400 dark:text-gray-500">
              <p className="text-center px-2">
                ⚠️ This tool is for research purposes only and does not provide
                medical advice. Always verify information with trusted sources.
              </p>
              <span className="hidden sm:inline">•</span>
              <p className="text-center whitespace-nowrap">
                Built with <span className="text-red-500">❤️</span> by{" "}
                <a
                  href="https://www.linkedin.com/in/Adarsh-Raj04"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="font-medium text-primary-600 dark:text-primary-400 hover:underline"
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
