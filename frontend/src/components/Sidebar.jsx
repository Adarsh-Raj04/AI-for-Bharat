import { useState, useEffect } from 'react'
import { useAuth0 } from '@auth0/auth0-react'
import api from '../services/api'

export default function Sidebar({ isOpen, currentSessionId, onSessionSelect, onNewSession }) {
  const [sessions, setSessions] = useState([])
  const [loading, setLoading] = useState(true)
  const { getAccessTokenSilently } = useAuth0()

  useEffect(() => {
    loadSessions()
  }, [])

  const loadSessions = async () => {
    try {
      const token = await getAccessTokenSilently()
      const response = await api.get('/sessions/', {
        headers: { Authorization: `Bearer ${token}` }
      })
      setSessions(response.data.sessions)
    } catch (error) {
      console.error('Failed to load sessions:', error)
    } finally {
      setLoading(false)
    }
  }

  const createNewSession = async () => {
    try {
      const token = await getAccessTokenSilently()
      const response = await api.post('/sessions/', 
        { session_name: 'New Chat' },
        { headers: { Authorization: `Bearer ${token}` } }
      )
      await loadSessions()
      if (onNewSession) {
        onNewSession(response.data.id)
      }
    } catch (error) {
      console.error('Failed to create session:', error)
    }
  }

  if (!isOpen) return null

  return (
    <aside className="fixed left-0 top-16 h-[calc(100vh-4rem)] w-64 bg-white border-r border-gray-200 overflow-y-auto">
      <div className="p-4">
        <button
          onClick={createNewSession}
          className="w-full flex items-center justify-center space-x-2 px-4 py-3 bg-primary-600 text-white rounded-lg hover:bg-primary-700 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span className="font-medium">New Chat</span>
        </button>
      </div>

      <div className="px-4 pb-4">
        <h2 className="text-xs font-semibold text-gray-500 uppercase tracking-wider mb-3">
          Chat History
        </h2>
        
        {loading ? (
          <div className="text-center py-8 text-gray-500">Loading...</div>
        ) : sessions.length === 0 ? (
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
                    ? 'bg-primary-50 border border-primary-200' 
                    : 'hover:bg-gray-100'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium truncate ${
                      currentSessionId === session.id ? 'text-primary-700' : 'text-gray-900'
                    }`}>
                      {session.session_name || 'Untitled Chat'}
                    </p>
                    <p className="text-xs text-gray-500">
                      {session.total_messages} messages
                    </p>
                  </div>
                  <svg className="w-4 h-4 text-gray-400 opacity-0 group-hover:opacity-100 transition-opacity" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    </aside>
  )
}
