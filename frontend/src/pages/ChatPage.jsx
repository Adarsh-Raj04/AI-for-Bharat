import { useState, useRef, useEffect } from "react";
import { useAuth0 } from "@auth0/auth0-react";
import api from "../services/api";
import ChatMessage from "../components/ChatMessage";
import DisclaimerModal from "../components/DisclaimerModal";

export default function ChatPage({ sessionId, onSessionChange }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [showDisclaimer, setShowDisclaimer] = useState(false);
  const messagesEndRef = useRef(null);
  const { getAccessTokenSilently } = useAuth0();

  useEffect(() => {
    checkDisclaimerStatus();
  }, []);

  // Load messages when session changes
  useEffect(() => {
    if (sessionId) {
      loadSessionMessages(sessionId);
    } else {
      setMessages([]);
    }
  }, [sessionId]);

  const loadSessionMessages = async (sid) => {
    try {
      const token = await getAccessTokenSilently();
      const response = await api.get(`/sessions/${sid}/messages`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      setMessages(response.data.messages);
    } catch (error) {
      console.error("Failed to load messages:", error);
      setMessages([]);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

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

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage = { role: "user", content: input };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setLoading(true);

    try {
      const token = await getAccessTokenSilently();
      const response = await api.post(
        "/chat/chat",
        {
          session_id: sessionId,
          message: input,
          stream: false,
        },
        {
          headers: { Authorization: `Bearer ${token}` },
        },
      );
      console.log("Chat response:", response.data);
      const assistantMessage = {
        role: "assistant",
        content: response.data.response.text,
        citations: response.data.response.citations,
        confidence: response.data.response.confidence,
        intent: response.data.response.intent,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      if (onSessionChange) {
        onSessionChange(response.data.session_id);
      }
    } catch (error) {
      console.error("Failed to send message:", error);
      const errorMessage = {
        role: "assistant",
        content:
          "Sorry, I encountered an error processing your request. Please try again.",
        error: true,
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      <div className="h-full flex flex-col bg-gray-50">
        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto px-4 py-6">
          <div className="max-w-4xl mx-auto space-y-6">
            {messages.length === 0 ? (
              <div className="text-center py-12">
                <svg
                  className="w-16 h-16 mx-auto text-gray-400 mb-4"
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
                <h2 className="text-2xl font-bold text-gray-900 mb-2">
                  Welcome to MedResearch AI
                </h2>
                <p className="text-gray-600 mb-6">
                  Your intelligent research assistant for life sciences
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 max-w-3xl mx-auto">
                  <ExampleQuery
                    icon="📄"
                    text="Summarize PMID 33301246"
                    onClick={() => setInput("Summarize PMID 33301246")}
                  />
                  <ExampleQuery
                    icon="⚖️"
                    text="Compare pembrolizumab vs nivolumab"
                    onClick={() =>
                      setInput("Compare pembrolizumab vs nivolumab for NSCLC")
                    }
                  />
                  <ExampleQuery
                    icon="📋"
                    text="FDA accelerated approval requirements"
                    onClick={() =>
                      setInput(
                        "What are FDA requirements for accelerated approval?",
                      )
                    }
                  />
                </div>
              </div>
            ) : (
              messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))
            )}

            {loading && (
              <div className="flex items-center space-x-2 text-gray-500">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600"></div>
                <span>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 bg-white px-4 py-4">
          <div className="max-w-4xl mx-auto">
            <form onSubmit={handleSendMessage} className="flex space-x-4">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about research papers, clinical trials, or regulatory guidance..."
                className="flex-1 px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                disabled={loading}
              />
              <button
                type="submit"
                disabled={loading || !input.trim()}
                className="px-6 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors font-medium"
              >
                Send
              </button>
            </form>
            <p className="text-xs text-gray-500 mt-2 text-center">
              ⚠️ For research purposes only. Not medical advice. Always verify
              information.
            </p>
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
      className="p-4 bg-white border border-gray-200 rounded-lg hover:border-primary-500 hover:shadow-md transition-all text-left"
    >
      <div className="text-2xl mb-2">{icon}</div>
      <p className="text-sm text-gray-700">{text}</p>
    </button>
  );
}
