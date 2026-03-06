import { useAuth0 } from '@auth0/auth0-react'
import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import LoginPage from './pages/LoginPage'
import LoadingSpinner from './components/LoadingSpinner'
import EmailVerificationRequired from './components/EmailVerificationRequired'
import { ThemeProvider } from './contexts/ThemeContext'
import api from './services/api'

function App() {
  const { isLoading, isAuthenticated, getAccessTokenSilently } = useAuth0()
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [authValidated, setAuthValidated] = useState(false)
  const [emailVerificationRequired, setEmailVerificationRequired] = useState(false)
  const [messages, setMessages] = useState([])   // ← lifted here

  useEffect(() => {
    const validateAuth = async () => {
      if (isAuthenticated) {
        try {
          const token = await getAccessTokenSilently()
          await api.get('/auth/me', {
            headers: { Authorization: `Bearer ${token}` }
          })
          setAuthValidated(true)
          setEmailVerificationRequired(false)
        } catch (error) {
          console.error('Failed to validate authentication:', error)
          if (error.response?.status === 403) {
            setEmailVerificationRequired(true)
          }
          setAuthValidated(false)
        }
      }
    }
    validateAuth()
  }, [isAuthenticated, getAccessTokenSilently])

  // Clear messages when session changes
  const handleSessionSelect = (sessionId) => {
    setCurrentSessionId(sessionId)
    setMessages([])
  }

  const handleNewSession = (sessionId) => {
    setCurrentSessionId(sessionId)
    setMessages([])
  }

  if (isLoading || (isAuthenticated && !authValidated && !emailVerificationRequired)) {
    return <LoadingSpinner />
  }

  if (isAuthenticated && emailVerificationRequired) {
    return <EmailVerificationRequired />
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
                  messages={messages}           // ← new
                  sessionId={currentSessionId}  // ← new
                >
                  <ChatPage
                    sessionId={currentSessionId}
                    onSessionChange={setCurrentSessionId}
                    onMessagesChange={setMessages}  // ← new
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
  )
}

export default App