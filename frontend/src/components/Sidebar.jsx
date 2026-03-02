import { useState, useEffect, useCallback } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import api from "../services/api";

export default function Sidebar({
  isOpen,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  isMobile,
}) {
  const [sessions, setSessions] = useState([]);
  const [loading, setLoading] = useState(true);

  const {
    getAccessTokenSilently,
    isAuthenticated,
    isLoading: authLoading,
  } = useAuth0();

  // ---------------------------------------
  // Load Sessions (SAFE VERSION)
  // ---------------------------------------
  const loadSessions = useCallback(async () => {
    try {
      setLoading(true);

      const token = await getAccessTokenSilently();

      const response = await api.get("/sessions/", {
        headers: { Authorization: `Bearer ${token}` },
      });

      const data = response?.data;

      // Supports BOTH:
      // 1️⃣ { sessions: [...] }
      // 2️⃣ [...]
      let parsedSessions = [];

      if (Array.isArray(data)) {
        parsedSessions = data;
      } else if (Array.isArray(data?.sessions)) {
        parsedSessions = data.sessions;
      }

      setSessions(parsedSessions);
    } catch (error) {
      console.error("Failed to load sessions:", error);
      setSessions([]); // prevent undefined state
    } finally {
      setLoading(false);
    }
  }, [getAccessTokenSilently]);

  // ---------------------------------------
  // Initial Load (wait for auth)
  // ---------------------------------------
  useEffect(() => {
    if (!authLoading && isAuthenticated) {
      loadSessions();
    }
  }, [authLoading, isAuthenticated, loadSessions]);

  // ---------------------------------------
  // Create New Session
  // ---------------------------------------
  const createNewSession = async () => {
    try {
      const token = await getAccessTokenSilently();

      const response = await api.post(
        "/sessions/",
        { session_name: "New Chat" },
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );

      await loadSessions();

      if (onNewSession) {
        onNewSession(response?.data?.id);
      }
    } catch (error) {
      console.error("Failed to create session:", error);
    }
  };

  // ---------------------------------------
  // Sidebar hidden
  // ---------------------------------------
  if (!isOpen) return null;

  // ---------------------------------------
  // UI
  // ---------------------------------------
  return (
    <aside className={`fixed left-0 top-[57px] sm:top-16 h-[calc(100vh-57px)] sm:h-[calc(100vh-4rem)] w-64 bg-white border-r border-gray-200 overflow-y-auto transition-transform duration-300 z-50 ${
      isOpen ? 'translate-x-0' : '-translate-x-full'
    } ${isMobile ? 'shadow-xl' : ''}`}>
      <div className="p-3 sm:p-4">
        <button
          onClick={createNewSession}
          className="w-full flex items-center justify-center space-x-2 px-3 sm:px-4 py-2.5 sm:py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors active:scale-95"
        >
          <svg
            className="w-5 h-5"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          <span className="font-medium text-sm sm:text-base">New Chat</span>
        </button>
      </div>

      <div className="px-3 sm:px-4 pb-4">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-2 sm:mb-3">
          Chat History
        </h2>

        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : !sessions?.length ? (
          <div className="text-center py-8 text-gray-500 text-sm">
            No chat history yet
          </div>
        ) : (
          <div className="space-y-2">
            {sessions.map((session) => (
              <button
                key={session.id}
                onClick={() => onSessionSelect && onSessionSelect(session.id)}
                className={`w-full text-left px-3 py-2 rounded-lg transition-colors group ${
                  currentSessionId === session.id
                    ? "bg-primary-50 border border-primary-200"
                    : "hover:bg-gray-100"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium truncate ${
                        currentSessionId === session.id
                          ? "text-primary-700"
                          : "text-gray-900"
                      }`}
                    >
                      {session.session_name || "Untitled Chat"}
                    </p>

                    <p className="text-xs text-gray-500">
                      {session.total_messages ?? 0} messages
                    </p>
                  </div>

                  <svg
                    className="w-4 h-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity"
                    fill="none"
                    stroke="currentColor"
                    viewBox="0 0 24 24"
                  >
                    <path
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      strokeWidth={2}
                      d="M9 5l7 7-7 7"
                    />
                  </svg>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </aside>
  );
}
