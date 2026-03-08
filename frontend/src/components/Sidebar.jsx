import { useState, useEffect, useCallback, useRef } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import { Trash2 } from "lucide-react";
import api from "../services/api";

export default function Sidebar({
  isOpen,
  currentSessionId,
  currentSessionName,
  onSessionSelect,
  onNewSession,
  isMobile,
  liveMessageCount,
  messagesLoading,
  onSessionNameChange, // ← called by ChatPage after auto-rename
}) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [renamingId, setRenamingId] = useState(null);
  const [renameValue, setRenameValue] = useState("");
  const [pendingSessionId, setPendingSessionId] = useState(null);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null); // ← inline delete confirm
  const renameInputRef = useRef(null);
  const [deletingId, setDeletingId] = useState(null);

  const {
    getAccessTokenSilently,
    isAuthenticated,
    isLoading: authLoading,
  } = useAuth0();

  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);
      const token = await getAccessTokenSilently();
      const response = await api.get("/sessions/", {
        headers: { Authorization: `Bearer ${token}` },
      });
      const data = response?.data;
      let parsed = [];
      if (Array.isArray(data)) parsed = data;
      else if (Array.isArray(data?.sessions)) parsed = data.sessions;
      setSessions(parsed);
    } catch (error) {
      console.error("Failed to load sessions:", error);
      setSessions([]);
    } finally {
      setLoading(false);
    }
  }, [getAccessTokenSilently]);

  useEffect(() => {
    if (!authLoading && isAuthenticated) loadSessions();
  }, [authLoading, isAuthenticated, loadSessions]);

  // Add newly created session OR update name if it changed (auto-rename)
  useEffect(() => {
    if (!currentSessionId) return;
    setSessions((prev) => {
      const exists = prev.some((s) => s.id === currentSessionId);
      if (exists) {
        // Update name in place if it changed
        return prev.map((s) =>
          s.id === currentSessionId && currentSessionName
            ? { ...s, session_name: currentSessionName }
            : s,
        );
      }
      return [
        {
          id: currentSessionId,
          session_name: currentSessionName || "New Chat",
          total_messages: liveMessageCount || 0,
        },
        ...prev,
      ];
    });
  }, [currentSessionId, currentSessionName]);

  // Live message count
  useEffect(() => {
    if (!currentSessionId) return;
    setSessions((prev) =>
      prev.map((s) =>
        s.id === currentSessionId
          ? { ...s, total_messages: liveMessageCount }
          : s,
      ),
    );
  }, [liveMessageCount, currentSessionId]);

  // Clear pending once session loaded
  useEffect(() => {
    if (
      pendingSessionId &&
      currentSessionId === pendingSessionId &&
      !messagesLoading
    ) {
      setPendingSessionId(null);
    }
  }, [currentSessionId, messagesLoading, pendingSessionId]);

  useEffect(() => {
    if (renamingId && renameInputRef.current) {
      renameInputRef.current.focus();
      renameInputRef.current.select();
    }
  }, [renamingId]);

  const createNewSession = () => {
    setPendingSessionId("new");
    setConfirmDeleteId(null);
    if (onNewSession) onNewSession(null);
  };

  const handleSessionClick = (sessionId) => {
    if (sessionId === currentSessionId) return;
    setConfirmDeleteId(null);
    setPendingSessionId(sessionId);
    onSessionSelect?.(sessionId);
  };

  // ── Rename ────────────────────────────────────────────────────────────────
  const startRename = (e, session) => {
    e.stopPropagation();
    setConfirmDeleteId(null);
    setRenamingId(session.id);
    setRenameValue(session.session_name || "");
  };

  const cancelRename = () => {
    setRenamingId(null);
    setRenameValue("");
  };

  const submitRename = async (sessionId) => {
    const trimmed = renameValue.trim();
    if (!trimmed) return cancelRename();
    try {
      const token = await getAccessTokenSilently();
      await api.patch(
        `/sessions/${sessionId}/rename`,
        { session_name: trimmed },
        { headers: { Authorization: `Bearer ${token}` } },
      );
      setSessions((prev) =>
        prev.map((s) =>
          s.id === sessionId ? { ...s, session_name: trimmed } : s,
        ),
      );
    } catch (error) {
      console.error("Failed to rename session:", error);
    } finally {
      cancelRename();
    }
  };

  const handleRenameKeyDown = (e, sessionId) => {
    if (e.key === "Enter") submitRename(sessionId);
    if (e.key === "Escape") cancelRename();
  };

  // ── Delete ────────────────────────────────────────────────────────────────
  const handleDeleteClick = (e, sessionId) => {
    e.stopPropagation();
    setRenamingId(null);
    setConfirmDeleteId((prev) => (prev === sessionId ? null : sessionId));
  };

  const confirmDelete = async (e, sessionId) => {
    e.stopPropagation();

    try {
      setDeletingId(sessionId); // ← start loading

      const token = await getAccessTokenSilently();

      await api.delete(`/sessions/${sessionId}`, {
        headers: { Authorization: `Bearer ${token}` },
      });

      // small UX delay so spinner is visible
      await new Promise((r) => setTimeout(r, 100));

      setSessions((prev) => prev.filter((s) => s.id !== sessionId));
      setConfirmDeleteId(null);

      if (currentSessionId === sessionId) {
        onNewSession?.(null);
      }
    } catch (error) {
      console.error("Failed to delete session:", error);
      setConfirmDeleteId(null);
    } finally {
      setDeletingId(null); // ← stop loading
    }
  };

  const cancelDelete = (e) => {
    e.stopPropagation();
    setConfirmDeleteId(null);
  };

  if (!isOpen) return null;

  const isSessionLoading = (sessionId) =>
    (pendingSessionId === sessionId || currentSessionId === sessionId) &&
    messagesLoading;

  return (
    <aside
      className={`fixed left-0 top-[57px] sm:top-16 h-[calc(100vh-57px)] sm:h-[calc(100vh-4rem)] w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 overflow-y-auto transition-transform duration-300 z-50 ${
        isOpen ? "translate-x-0" : "-translate-x-full"
      } ${isMobile ? "shadow-xl" : ""}`}
    >
      {/* New Chat */}
      <div className="p-3 sm:p-4">
        <button
          onClick={createNewSession}
          disabled={pendingSessionId === "new" && messagesLoading}
          className="w-full flex items-center justify-center gap-2 px-3 sm:px-4 py-2.5 sm:py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-60 transition-colors font-medium"
        >
          {pendingSessionId === "new" && messagesLoading ? (
            <>
              <span className="inline-block w-3.5 h-3.5 border-2 border-white border-t-transparent rounded-full animate-spin" />
              Creating…
            </>
          ) : (
            "New Chat"
          )}
        </button>
      </div>

      {/* Session list */}
      <div className="px-3 sm:px-4 pb-4">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 sm:mb-3">
          Chat History
        </h2>

        {loading ? (
          <div className="text-center py-8 text-gray-500 text-sm">Loading…</div>
        ) : !sessions.length ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No chat history yet
          </div>
        ) : (
          <div className="space-y-1">
            {sessions.map((session) => {
              const isActive = currentSessionId === session.id;
              const isRenaming = renamingId === session.id;
              const isConfirmingDelete = confirmDeleteId === session.id;
              const isFetching = isSessionLoading(session.id);
              const isDeleting = deletingId === session.id;

              return (
                <div
                  key={session.id}
                  className={`rounded-lg group transition-colors ${
                    isActive
                      ? "bg-primary-50 dark:bg-primary-900/20 border border-primary-200 dark:border-primary-700"
                      : "hover:bg-gray-100 dark:hover:bg-gray-700"
                  }`}
                >
                  {/* ── Rename input ── */}
                  {isRenaming ? (
                    <div className="flex items-center gap-1 px-2 py-1.5">
                      <input
                        ref={renameInputRef}
                        value={renameValue}
                        onChange={(e) => setRenameValue(e.target.value)}
                        onKeyDown={(e) => handleRenameKeyDown(e, session.id)}
                        onBlur={() => submitRename(session.id)}
                        className="flex-1 text-sm px-2 py-1 rounded border dark:bg-gray-700 dark:border-gray-600 dark:text-gray-100"
                      />
                    </div>
                  ) : /* ── Delete confirm ── */
                  isConfirmingDelete ? (
                    deletingId === session.id ? (
                      <div
                        className="px-3 py-2 flex items-center gap-2 text-xs text-gray-500"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span className="flex-shrink-0 w-3 h-3 border-2 border-primary-400 border-t-transparent rounded-full animate-spin" />
                        Deleting…
                      </div>
                    ) : (
                      <div
                        className="px-3 py-2"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <p className="text-xs text-gray-600 dark:text-gray-300 mb-2 font-medium">
                          Delete this chat?
                        </p>
                        <div className="flex gap-2">
                          <button
                            onClick={(e) => confirmDelete(e, session.id)}
                            className="flex-1 py-1 rounded text-xs font-medium bg-red-500 hover:bg-red-600 text-white transition-colors"
                          >
                            Delete
                          </button>
                          <button
                            onClick={cancelDelete}
                            className="flex-1 py-1 rounded text-xs font-medium bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 transition-colors"
                          >
                            Cancel
                          </button>
                        </div>
                      </div>
                    )
                  ) : (
                    /* ── Normal row ── */
                    <button
                      onClick={() => handleSessionClick(session.id)}
                      disabled={isFetching || isDeleting}
                      className="w-full text-left px-3 py-2 flex items-center justify-between"
                    >
                      <div className="flex-1 min-w-0 flex items-center gap-2">
                        {isFetching || isDeleting ? (
                          <span className="flex-shrink-0 w-3 h-3 border-2 border-primary-400 border-t-transparent rounded-full animate-spin" />
                        ) : isActive ? (
                          <span className="flex-shrink-0 w-1.5 h-1.5 rounded-full bg-primary-500" />
                        ) : (
                          <span className="flex-shrink-0 w-1.5 h-1.5" />
                        )}
                        <div className="min-w-0">
                          <p
                            className={`text-xs font-medium truncate ${isActive ? "text-primary-700 dark:text-primary-300" : "text-gray-800 dark:text-gray-200"}`}
                            title={session.session_name || "Untitled Chat"}
                          >
                            {session.session_name || "Untitled Chat"}
                          </p>
                          <p className="text-xs text-gray-500 dark:text-gray-400">
                            {isFetching
                              ? "Loading…"
                              : `${session.total_messages ?? 0} messages`}
                          </p>
                        </div>
                      </div>

                      {/* Hover actions — rename + delete */}
                      {!isFetching && !isDeleting && (
                        <div className="opacity-0 group-hover:opacity-100 flex items-center gap-1 ml-1 flex-shrink-0 transition-opacity">
                          <span
                            role="button"
                            onClick={(e) => startRename(e, session)}
                            title="Rename"
                            className="p-0.5 text-gray-400 hover:text-primary-600 dark:hover:text-primary-400"
                          >
                            ✏️
                          </span>
                          <span
                            role="button"
                            onClick={(e) => handleDeleteClick(e, session.id)}
                            title="Delete"
                            className="p-0.5 text-gray-400 hover:text-red-500 dark:hover:text-red-400"
                          >
                            <Trash2 className="w-3.5 h-3.5" />
                          </span>
                        </div>
                      )}
                    </button>
                  )}
                </div>
              );
            })}
          </div>
        )}
      </div>
    </aside>
  );
}
