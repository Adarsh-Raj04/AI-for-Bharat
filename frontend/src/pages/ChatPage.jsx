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
      <div className="h-full flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto px-3 sm:px-4 py-4 sm:py-6">
          <div className="max-w-4xl mx-auto space-y-4 sm:space-y-6">
            {messages.length === 0 ? (
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
              messages.map((message, index) => (
                <ChatMessage key={index} message={message} />
              ))
            )}

            {loading && (
              <div className="flex items-center space-x-2 text-gray-500 dark:text-gray-400">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-primary-600 dark:border-primary-400"></div>
                <span>Thinking...</span>
              </div>
            )}

            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Input Area */}
        <div className="border-t border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-3 sm:px-4 py-3 sm:py-4 safe-area-bottom transition-colors">
          <div className="max-w-4xl mx-auto">
            <form
              onSubmit={handleSendMessage}
              className="flex flex-col sm:flex-row gap-2 sm:gap-4"
            >
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about research papers..."
                className="flex-1 px-3 sm:px-4 py-2.5 sm:py-3 text-sm sm:text-base border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 dark:focus:ring-primary-400 focus:border-transparent transition-colors"
                disabled={loading}
              />
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
    ⚠️ This tool is for research purposes only and does not provide medical advice. Always verify information with trusted sources.
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
      <p className="text-xs sm:text-sm text-gray-700 dark:text-gray-300">{text}</p>
    </button>
  );
}
