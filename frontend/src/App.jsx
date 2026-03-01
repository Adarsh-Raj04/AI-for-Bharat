import { useAuth0 } from '@auth0/auth0-react'
import { useState, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import Layout from './components/Layout'
import ChatPage from './pages/ChatPage'
import LoginPage from './pages/LoginPage'
import LoadingSpinner from './components/LoadingSpinner'
import EmailVerificationRequired from './components/EmailVerificationRequired'
import api from './services/api'

function App() {
  const { isLoading, isAuthenticated, getAccessTokenSilently } = useAuth0()
  const [currentSessionId, setCurrentSessionId] = useState(null)
  const [authValidated, setAuthValidated] = useState(false)
  const [emailVerificationRequired, setEmailVerificationRequired] = useState(false)

  // Validate authentication with backend on mount and page refresh
  useEffect(() => {
    const validateAuth = async () => {
      if (isAuthenticated) {
        try {
          const token = await getAccessTokenSilently()
          // Call backend to validate token and get user info
          await api.get('/auth/me', {
            headers: { Authorization: `Bearer ${token}` }
          })
          setAuthValidated(true)
          setEmailVerificationRequired(false)
        } catch (error) {
          console.error('Failed to validate authentication:', error)
          
          // Check if it's an email verification error
          if (error.response?.status === 403) {
            setEmailVerificationRequired(true)
          }
          
          setAuthValidated(false)
        }
      }
    }

    validateAuth()
  }, [isAuthenticated, getAccessTokenSilently])

  if (isLoading || (isAuthenticated && !authValidated && !emailVerificationRequired)) {
    return <LoadingSpinner />
  }

  // Show email verification screen if required
  if (isAuthenticated && emailVerificationRequired) {
    return <EmailVerificationRequired />
  }

  const handleSessionSelect = (sessionId) => {
    setCurrentSessionId(sessionId)
  }

  const handleNewSession = (sessionId) => {
    setCurrentSessionId(sessionId)
  }

  return (
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
              >
                <ChatPage 
                  sessionId={currentSessionId}
                  onSessionChange={setCurrentSessionId}
                />
              </Layout>
            ) : (
              <Navigate to="/login" replace />
            )
          }
        />
      </Routes>
    </Router>
  )
}

export default App
