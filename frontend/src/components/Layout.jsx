import { useState, useEffect } from "react";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import React from "react";

export default function Layout({
  children,
  currentSessionId,
  onSessionSelect,
  onNewSession,
  messages,
  sessionId,
  currentSessionName, // ← NEW: threaded from App → Sidebar
  onSessionNameChange, // ← NEW: threaded from App → Sidebar
  messagesLoading, // ← NEW: threaded from App → Sidebar
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const messageCount = messages?.length || 0;

  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth < 768;
      setIsMobile(mobile);
      setSidebarOpen(!mobile);
    };
    checkMobile();
    window.addEventListener("resize", checkMobile);
    return () => window.removeEventListener("resize", checkMobile);
  }, []);

  const handleToggleSidebar = () => setSidebarOpen(!sidebarOpen);

  // Add this inside Layout.jsx if you want extra-smooth focus transitions
  useEffect(() => {
    if (!sidebarOpen || !isMobile) {
      // Small delay to allow layout transition to finish
      setTimeout(() => {
        document.querySelector('input[type="text"]')?.focus();
      }, 100);
    }
  }, [sidebarOpen, isMobile]);

  const handleSessionSelect = (sid) => {
    onSessionSelect?.(sid);
    if (isMobile) setSidebarOpen(false);
  };

  const handleNewSession = (sid) => {
    onNewSession?.(sid);
    if (isMobile) setSidebarOpen(false);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
      <Navbar
        onToggleSidebar={handleToggleSidebar}
        sidebarOpen={sidebarOpen}
        messages={messages}
        sessionId={sessionId}
      />

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
          currentSessionName={currentSessionName} // ← NEW
          onSessionSelect={handleSessionSelect}
          onNewSession={handleNewSession}
          isMobile={isMobile}
          liveMessageCount={messageCount}
          messagesLoading={messagesLoading} // ← NEW
          onSessionNameChange={onSessionNameChange} // ← NEW
        />

        <main
          className={`flex-1 overflow-hidden transition-all duration-300 ${
            !isMobile && sidebarOpen ? "md:ml-64" : "ml-0"
          }`}
        >
          {children}
        </main>
      </div>
    </div>
  );
}
