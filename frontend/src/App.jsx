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
import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

function App() {
  const { isLoading, isAuthenticated, getAccessTokenSilently } = useAuth0();
  const [authValidated, setAuthValidated] = useState(false);
  const [emailVerificationRequired, setEmailVerificationRequired] =
    useState(false);
  const [messages, setMessages] = useState([]);
  const [currentSessionName, setCurrentSessionName] = useState(null);
  const [messagesLoading, setMessagesLoading] = useState(false);

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

  if (
    isLoading ||
    (isAuthenticated && !authValidated && !emailVerificationRequired)
  ) {
    return <LoadingSpinner />;
  }

  if (isAuthenticated && emailVerificationRequired) {
    return <EmailVerificationRequired />;
  }

  const chatElement = (
    <ChatPage
      onMessagesChange={setMessages}
      onSessionNameChange={setCurrentSessionName}
      onMessagesLoadingChange={setMessagesLoading}
    />
  );

  return (
    <ThemeProvider>
      <Router>
        <ToastContainer position="top-center" autoClose={3000} />
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route
            path="/"
            element={
              isAuthenticated ? (
                <Layout
                  messages={messages}
                  currentSessionName={currentSessionName}
                  onSessionNameChange={setCurrentSessionName}
                  messagesLoading={messagesLoading}
                />
              ) : (
                <Navigate to="/login" replace />
              )
            }
          >
            <Route index element={chatElement} />
            <Route path="chat/:sessionId" element={chatElement} />
          </Route>
        </Routes>
      </Router>
    </ThemeProvider>
  );
}

export default App;
