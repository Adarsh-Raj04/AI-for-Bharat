import { createContext, useContext, useState, useEffect } from 'react'

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
  const [isDark, setIsDark] = useState(() => {
    // Check localStorage or system preference
    const saved = localStorage.getItem('theme')
    console.log('Initial theme from localStorage:', saved)
    if (saved) {
      return saved === 'dark'
    }
    const systemPrefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches
    console.log('System prefers dark:', systemPrefersDark)
    return systemPrefersDark
  })

  useEffect(() => {
    // Update localStorage and document class
    console.log('Theme changed to:', isDark ? 'dark' : 'light')
    localStorage.setItem('theme', isDark ? 'dark' : 'light')
    if (isDark) {
      document.documentElement.classList.add('dark')
      console.log('Added dark class, current classes:', document.documentElement.className)
    } else {
      document.documentElement.classList.remove('dark')
      console.log('Removed dark class, current classes:', document.documentElement.className)
    }
  }, [isDark])

  const toggleTheme = () => {
    console.log('Toggle theme clicked, current:', isDark)
    setIsDark(!isDark)
  }

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  const context = useContext(ThemeContext)
  if (!context) {
    throw new Error('useTheme must be used within ThemeProvider')
  }
  return context
}
