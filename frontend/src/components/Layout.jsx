import { useState, useEffect } from 'react'
import Navbar from './Navbar'
import Sidebar from './Sidebar'

export default function Layout({
  children,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  messages,   // ← new: from App.jsx
  sessionId,  // ← new: from App.jsx
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768
      setIsMobile(mobile)
      setSidebarOpen(!mobile)
    }
    checkMobile()
    window.addEventListener('resize', checkMobile)
    return () => window.removeEventListener('resize', checkMobile)
  }, [])

  const handleToggleSidebar = () => setSidebarOpen(!sidebarOpen)

  const handleSessionSelect = (sid) => {
    onSessionSelect?.(sid)
    if (isMobile) setSidebarOpen(false)
  }

  const handleNewSession = (sid) => {
    onNewSession?.(sid)
    if (isMobile) setSidebarOpen(false)
  }

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
      <Navbar
        onToggleSidebar={handleToggleSidebar}
        sidebarOpen={sidebarOpen}
        messages={messages}    // ← new
        sessionId={sessionId}  // ← new
      />

      {/* Mobile overlay */}
      {isMobile && sidebarOpen && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-40 md:hidden"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <div className="flex flex-1 overflow-hidden">
        <Sidebar
          isOpen={sidebarOpen}
          currentSessionId={currentSessionId}
          onSessionSelect={handleSessionSelect}
          onNewSession={handleNewSession}
          isMobile={isMobile}
        />

        <main className={`flex-1 overflow-hidden transition-all duration-300 ${
          !isMobile && sidebarOpen ? 'md:ml-64' : 'ml-0'
        }`}>
          {children}
        </main>
      </div>
    </div>
  )
}