import { useState, useEffect } from "react";

/**
 * ExportModal
 * Props:
 *   isOpen        – boolean
 *   onClose       – () => void
 *   messages      – full message array from ChatPage (each assistant msg must have an `id` field)
 *   sessionId     – current session id
 *   getToken      – async () => token  (pass getAccessTokenSilently)
 *   apiInstance   – the axios `api` instance
 *
 * Backend endpoint:
 *   POST /export?message_id=<id>&format=<fmt>   ← single message
 *   POST /export?session_id=<id>&format=<fmt>   ← full session
 *   (no request body needed)
 */
export default function ExportModal({
  isOpen,
  onClose,
  messages,
  sessionId,
  getToken,
  apiInstance,
}) {
  const [scope, setScope] = useState("session"); // "session" | "single"
  const [selectedMsgIndex, setSelectedMsgIndex] = useState(null);
  const [format, setFormat] = useState("pdf"); // "pdf" | "markdown"
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // Only non-error assistant messages are selectable
  const assistantMessages = messages
    .map((m, i) => ({ ...m, originalIndex: i }))
    .filter((m) => m.role === "assistant" && !m.error);

  // Auto-select last assistant message whenever modal opens
  useEffect(() => {
    if (isOpen) {
      if (assistantMessages.length > 0) {
        setSelectedMsgIndex(
          assistantMessages[assistantMessages.length - 1].originalIndex
        );
      }
      setError("");
    }
  }, [isOpen]);

  if (!isOpen) return null;

  const handleExport = async () => {
    setError("");

    if (scope === "single" && selectedMsgIndex === null) {
      setError("Please select a message to export.");
      return;
    }

    setLoading(true);
    try {
      const token = await getToken();

      // Single endpoint: POST /export
      // Params go on the query string — no body needed
      let params;
      if (scope === "session") {
        params = { session_id: sessionId, format };
      } else {
        const msg = messages[selectedMsgIndex];
        // msg.id is the DB message id stored from the chat API response
        if (!msg?.id) {
          setError(
            "This message has no ID. Make sure ChatPage stores the `id` field from the chat API response."
          );
          setLoading(false);
          return;
        }
        params = { message_id: msg.id, format };
      }

      const response = await apiInstance.post("/chat/export", null, {
        headers: { Authorization: `Bearer ${token}` },
        params,           // axios serialises this as ?session_id=...&format=...
        responseType: "blob",
      });

      // Trigger browser download
      const ext = format === "pdf" ? "pdf" : "md";
      const mimeType = format === "pdf" ? "application/pdf" : "text/markdown";
      const blob = new Blob([response.data], { type: mimeType });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `medresearch-export-${Date.now()}.${ext}`;
      a.click();
      URL.revokeObjectURL(url);
      onClose();
    } catch (err) {
      console.error("Export failed:", err);
      setError("Export failed. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  const truncate = (str, n = 80) =>
    str && str.length > n ? str.slice(0, n) + "…" : str;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/50 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Modal */}
      <div className="relative w-full max-w-md bg-white dark:bg-gray-800 rounded-2xl shadow-2xl border border-gray-200 dark:border-gray-700 overflow-hidden animate-in">

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-100 dark:border-gray-700">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-primary-100 dark:bg-primary-900/40 rounded-lg flex items-center justify-center">
              <svg className="w-4 h-4 text-primary-600 dark:text-primary-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
            </div>
            <h2 className="text-base font-semibold text-gray-900 dark:text-gray-100">Export</h2>
          </div>
          <button
            onClick={onClose}
            className="w-8 h-8 flex items-center justify-center rounded-lg text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        <div className="px-6 py-5 space-y-5">

          {/* Step 1: Scope */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">
              What to export
            </p>
            <div className="grid grid-cols-2 gap-2">
              <ScopeCard
                active={scope === "session"}
                onClick={() => setScope("session")}
                icon={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                  </svg>
                }
                label="Full Session"
                desc="All messages in this conversation"
              />
              <ScopeCard
                active={scope === "single"}
                onClick={() => setScope("single")}
                icon={
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                }
                label="Single Response"
                desc="One AI response of your choice"
              />
            </div>
          </div>

          {/* Step 2: Message selector (only for single) */}
          {scope === "single" && (
            <div>
              <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">
                Select a response
              </p>
              {assistantMessages.length === 0 ? (
                <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                  No AI responses in this session yet.
                </p>
              ) : (
                <div className="space-y-2 max-h-44 overflow-y-auto pr-1 scrollbar-thin">
                  {assistantMessages.map((msg, idx) => (
                    <button
                      key={msg.originalIndex}
                      onClick={() => setSelectedMsgIndex(msg.originalIndex)}
                      className={`w-full text-left px-3 py-2.5 rounded-lg border text-sm transition-all ${
                        selectedMsgIndex === msg.originalIndex
                          ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20 text-primary-700 dark:text-primary-300"
                          : "border-gray-200 dark:border-gray-600 bg-gray-50 dark:bg-gray-700/50 text-gray-700 dark:text-gray-300 hover:border-gray-300 dark:hover:border-gray-500"
                      }`}
                    >
                      <div className="flex items-start gap-2">
                        <span className={`mt-0.5 w-5 h-5 rounded-full flex-shrink-0 flex items-center justify-center text-xs font-bold ${
                          selectedMsgIndex === msg.originalIndex
                            ? "bg-primary-500 text-white"
                            : "bg-gray-200 dark:bg-gray-600 text-gray-500 dark:text-gray-400"
                        }`}>
                          {idx + 1}
                        </span>
                        <span className="leading-relaxed line-clamp-2">
                          {truncate(msg.content)}
                        </span>
                      </div>
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          {/* Step 3: Format */}
          <div>
            <p className="text-xs font-semibold uppercase tracking-wider text-gray-400 dark:text-gray-500 mb-3">
              Export format
            </p>
            <div className="grid grid-cols-2 gap-2">
              <FormatCard
                active={format === "pdf"}
                onClick={() => setFormat("pdf")}
                icon="📄"
                label="PDF"
                desc="Formatted document"
              />
              <FormatCard
                active={format === "markdown"}
                onClick={() => setFormat("markdown")}
                icon="📝"
                label="Markdown"
                desc="Plain text with markup"
              />
            </div>
          </div>

          {/* Error */}
          {error && (
            <p className="text-sm text-red-600 dark:text-red-400 bg-red-50 dark:bg-red-900/20 px-3 py-2 rounded-lg">
              {error}
            </p>
          )}
        </div>

        {/* Footer */}
        <div className="px-6 pb-5 flex gap-3">
          <button
            onClick={onClose}
            className="flex-1 px-4 py-2.5 rounded-lg border border-gray-200 dark:border-gray-600 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={handleExport}
            disabled={loading || (scope === "single" && selectedMsgIndex === null)}
            className="flex-1 px-4 py-2.5 rounded-lg bg-primary-600 dark:bg-primary-700 text-white text-sm font-medium hover:bg-primary-700 dark:hover:bg-primary-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
          >
            {loading ? (
              <>
                <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Exporting…
              </>
            ) : (
              <>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                </svg>
                Export
              </>
            )}
          </button>
        </div>
      </div>

      <style>{`
        @keyframes animate-in {
          from { opacity: 0; transform: scale(0.95) translateY(8px); }
          to   { opacity: 1; transform: scale(1) translateY(0); }
        }
        .animate-in { animation: animate-in 0.18s ease-out both; }
        .scrollbar-thin { scrollbar-width: thin; }
      `}</style>
    </div>
  );
}

function ScopeCard({ active, onClick, icon, label, desc }) {
  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-xl border text-left transition-all ${
        active
          ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
          : "border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500 bg-white dark:bg-gray-700/40"
      }`}
    >
      <div className={`mb-1.5 ${active ? "text-primary-600 dark:text-primary-400" : "text-gray-400 dark:text-gray-500"}`}>
        {icon}
      </div>
      <p className={`text-sm font-medium ${active ? "text-primary-700 dark:text-primary-300" : "text-gray-700 dark:text-gray-300"}`}>
        {label}
      </p>
      <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{desc}</p>
    </button>
  );
}

function FormatCard({ active, onClick, icon, label, desc }) {
  return (
    <button
      onClick={onClick}
      className={`p-3 rounded-xl border text-left transition-all ${
        active
          ? "border-primary-500 bg-primary-50 dark:bg-primary-900/20"
          : "border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500 bg-white dark:bg-gray-700/40"
      }`}
    >
      <div className="text-xl mb-1">{icon}</div>
      <p className={`text-sm font-medium ${active ? "text-primary-700 dark:text-primary-300" : "text-gray-700 dark:text-gray-300"}`}>
        {label}
      </p>
      <p className="text-xs text-gray-400 dark:text-gray-500 mt-0.5">{desc}</p>
    </button>
  );
}