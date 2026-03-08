import { useAuth0 } from "@auth0/auth0-react";
import { useState, useEffect } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  Navigate,
} from "react-router-dom";
import Layout from "./components/Layout";
import ChatPage from "./pages/ChatPage";
import LoginPage from "./pages/LoginPage";
import LoadingSpinner from "./components/LoadingSpinner";
import EmailVerificationRequired from "./components/EmailVerificationRequired";
import { ThemeProvider } from "./contexts/ThemeContext";
import api from "./services/api";

function App() {
  const { isLoading, isAuthenticated, getAccessTokenSilently } = useAuth0();
  const [currentSessionId, setCurrentSessionId] = useState(null);
  const [authValidated, setAuthValidated] = useState(false);
  const [emailVerificationRequired, setEmailVerificationRequired] =
    useState(false);
  const [messages, setMessages] = useState([]);
  const [currentSessionName, setCurrentSessionName] = useState(null);
  const [messagesLoading, setMessagesLoading] = useState(false);
  const [newChatTrigger, setNewChatTrigger] = useState(0); // ← incremented to force ChatPage reset

  useEffect(() => {
    const validateAuth = async () => {
      if (isAuthenticated) {
        try {
          const token = await getAccessTokenSilently();
          await api.get("/auth/me", {
            headers: { Authorization: `Bearer ${token}` },
          });
          setAuthValidated(true);
          setEmailVerificationRequired(false);
        } catch (error) {
          console.error("Failed to validate authentication:", error);
          if (error.response?.status === 403) {
            setEmailVerificationRequired(true);
          }
          setAuthValidated(false);
        }
      }
    };
    validateAuth();
  }, [isAuthenticated, getAccessTokenSilently]);

  const handleSessionSelect = (sessionId) => {
    setCurrentSessionId(sessionId);
    setMessages([]);
    setCurrentSessionName(null); // ← reset name on switch
  };

  const handleNewSession = (sessionId) => {
    setCurrentSessionId(sessionId);
    setMessages([]);
    setCurrentSessionName(null);
    setNewChatTrigger((prev) => prev + 1); // ← signal ChatPage to reset even if sessionId stays null
  };

  if (
    isLoading ||
    (isAuthenticated && !authValidated && !emailVerificationRequired)
  ) {
    return <LoadingSpinner />;
  }

  if (isAuthenticated && emailVerificationRequired) {
    return <EmailVerificationRequired />;
  }

  return (
    <ThemeProvider>
      <Router>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/"
            element={
              isAuthenticated ? (
                <Layout
                  currentSessionId={currentSessionId}
                  onSessionSelect={handleSessionSelect}
                  onNewSession={handleNewSession}
                  messages={messages}
                  sessionId={currentSessionId}
                  currentSessionName={currentSessionName}
                  onSessionNameChange={setCurrentSessionName}
                  messagesLoading={messagesLoading}
                  newChatTrigger={newChatTrigger}
                >
                  <ChatPage
                    sessionId={currentSessionId}
                    onSessionChange={setCurrentSessionId}
                    onMessagesChange={setMessages}
                    onSessionNameChange={setCurrentSessionName}
                    onMessagesLoadingChange={setMessagesLoading}
                    newChatTrigger={newChatTrigger}
                  />
                </Layout>
              ) : (
                <Navigate to="/login" replace />
              )
            }
          />
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
