import { useState, useEffect } from "react";
import { Outlet, useNavigate, useParams } from "react-router-dom";
import Navbar from "./Navbar";
import Sidebar from "./Sidebar";
import React from "react";

export default function Layout({
  messages,
  currentSessionName,
  onSessionNameChange,
  messagesLoading,
}) {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isMobile, setIsMobile] = useState(false);
  const messageCount = messages?.length || 0;
  const navigate = useNavigate();
  const { sessionId: currentSessionId } = useParams();

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

  useEffect(() => {
    if (!sidebarOpen || !isMobile) {
      setTimeout(() => {
        document.querySelector('input[type="text"]')?.focus();
      }, 100);
    }
  }, [sidebarOpen, isMobile]);

  const handleSessionSelect = (sid) => {
    navigate(`/chat/${sid}`);
    if (isMobile) setSidebarOpen(false);
  };

  const handleNewSession = () => {
    navigate("/");
    if (isMobile) setSidebarOpen(false);
  };

  return (
    <div className="h-screen flex flex-col bg-gray-50 dark:bg-gray-900 transition-colors">
      <Navbar
        onToggleSidebar={handleToggleSidebar}
        sidebarOpen={sidebarOpen}
        messages={messages}
        sessionId={currentSessionId}
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
          currentSessionName={currentSessionName}
          onSessionSelect={handleSessionSelect}
          onNewSession={handleNewSession}
          isMobile={isMobile}
          liveMessageCount={messageCount}
          messagesLoading={messagesLoading}
          onSessionNameChange={onSessionNameChange}
        />

        <main
          className={`flex-1 overflow-hidden transition-all duration-300 ${
            !isMobile && sidebarOpen ? "md:ml-64" : "ml-0"
          }`}
        >
          <Outlet />
        </main>
      </div>
    </div>
  );
}
