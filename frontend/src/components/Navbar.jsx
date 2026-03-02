import { useState } from 'react'
import { useAuth0 } from '@auth0/auth0-react'

export default function Navbar({ onToggleSidebar, sidebarOpen }) {
  const { user, logout } = useAuth0()
  const [showMenu, setShowMenu] = useState(false)
  const username = user?.["https://medresearch-ai-api/username"] || user?.name || user?.nickname

  return (
    <nav className="bg-white border-b border-gray-200 px-3 sm:px-4 py-3 flex items-center justify-between relative z-50">
      <div className="flex items-center space-x-2 sm:space-x-4">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg hover:bg-gray-100 transition-colors"
          aria-label="Toggle sidebar"
        >
          {sidebarOpen ? (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
        
        <div className="flex items-center space-x-2">
          <svg className="w-6 h-6 sm:w-8 sm:h-8 text-primary-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
          <h1 className="text-base sm:text-xl font-bold text-gray-900 hidden xs:block">MedResearch AI</h1>
          <h1 className="text-base font-bold text-gray-900 xs:hidden">MedAI</h1>
        </div>
      </div>

      <div className="flex items-center space-x-2 sm:space-x-4">
        {/* Desktop menu */}
        <div className="hidden sm:flex items-center space-x-4">
          <div className="text-sm text-gray-700">
            <span className="font-medium">{username}</span>
          </div>
          
          <button
            onClick={() => logout({ logoutParams: { returnTo: window.location.origin } })}
            className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-100 rounded-lg transition-colors"
          >
            Logout
          </button>
        </div>

        {/* Mobile menu button */}
        <button
          onClick={() => setShowMenu(!showMenu)}
          className="sm:hidden p-2 rounded-lg hover:bg-gray-100 transition-colors"
          aria-label="User menu"
        >
          <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        </button>

        {/* Mobile dropdown menu */}
        {showMenu && (
          <div className="absolute top-full right-3 mt-2 w-48 bg-white rounded-lg shadow-lg border border-gray-200 py-2 sm:hidden">
            <div className="px-4 py-2 border-b border-gray-200">
              <p className="text-sm font-medium text-gray-900">{username}</p>
              <p className="text-xs text-gray-500">{user?.email}</p>
            </div>
            <button
              onClick={() => {
                setShowMenu(false)
                logout({ logoutParams: { returnTo: window.location.origin } })
              }}
              className="w-full text-left px-4 py-2 text-sm text-gray-700 hover:bg-gray-100 transition-colors"
            >
              Logout
            </button>
          </div>
        )}
      </div>
    </nav>
  )
}
