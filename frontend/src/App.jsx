import React, { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import { FiMoon, FiSend, FiSun } from 'react-icons/fi'
import { GiChefToque } from 'react-icons/gi'
import ChatSidebar from './components/ChatSidebar'
import { addMessageToChat, createNewChat, getChatById } from './utils/chatStorage'

const ThemeContext = createContext()

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() => {
    return localStorage.getItem('reseppai-theme') || 'light'
  })

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    localStorage.setItem('reseppai-theme', theme)
  }, [theme])

  const toggleTheme = () => {
    setTheme(prev => prev === 'light' ? 'dark' : 'light')
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  )
}

export function useTheme() {
  return useContext(ThemeContext)
}

const API_URL = import.meta.env.VITE_API_URL || '/api'

function normalizeMessages(messages = []) {
  return Array.isArray(messages)
    ? messages.map(({ role, content, sources }) => ({ role, content, ...(sources ? { sources } : {}) }))
    : []
}

function loadMessages() {
  try {
    return JSON.parse(localStorage.getItem('reseppai-messages') || '[]')
  } catch {
    return []
  }
}

function saveMessages(messages) {
  localStorage.setItem('reseppai-messages', JSON.stringify(messages.slice(-100)))
}

function generateSessionId() {
  return 'sess_' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8)
}

function loadSessionId() {
  let id = localStorage.getItem('reseppai-session')
  if (!id) {
    id = generateSessionId()
    localStorage.setItem('reseppai-session', id)
  }
  return id
}

function Header() {
  const { theme, toggleTheme } = useTheme()
  return (
    <header className="chat-header">
      <div className="header-left">
        <span className="header-logo"><GiChefToque className="icon-sm" /></span>
        <h1 className="header-title">ResepAI</h1>
      </div>
      <button className="theme-toggle" onClick={toggleTheme} aria-label="Toggle theme">
        {theme === 'light' ? <FiMoon className="icon-sm" /> : <FiSun className="icon-sm" />}
      </button>
    </header>
  )
}

function MessageBubble({ message }) {
  const isUser = message.role === 'user'
  return (
    <div className={`message-bubble ${isUser ? 'user' : 'bot'}`}>
      {!isUser && <div className="message-avatar"><GiChefToque className="icon-sm" /></div>}
      <div className="message-content">
        <div className="message-text">
          {isUser ? message.content : (
            <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeSanitize]}>
              {message.content}
            </ReactMarkdown>
          )}
        </div>
      </div>
      {isUser && <div className="message-avatar user-avatar">👤</div>}
    </div>
  )
}

function TypingIndicator() {
  return (
    <div className="message-bubble bot">
      <div className="message-avatar"><GiChefToque className="icon-sm" /></div>
      <div className="message-content">
        <div className="typing-indicator">
          <span></span><span></span><span></span>
        </div>
      </div>
    </div>
  )
}

function MessageList({ messages, isLoading }) {
  const bottomRef = useRef(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  return (
    <div className="message-list">
      {messages.length === 0 && (
        <div className="welcome-screen">
          <div className="welcome-icon"><GiChefToque /></div>
          <h2>Selamat datang di ResepAI!</h2>
          <p>Tanyakan resep masakan Indonesia, bahan yang kamu punya, atau cara memasak.</p>
          <div className="welcome-examples">
            <button className="example-chip" onClick={() => window.dispatchEvent(new CustomEvent('sendMessage', { detail: 'Resep ayam goreng?' }))}>
              Resep ayam goreng?
            </button>
            <button className="example-chip" onClick={() => window.dispatchEvent(new CustomEvent('sendMessage', { detail: 'Masakan dengan tahu dan tempe' }))}>
              Masakan dengan tahu dan tempe
            </button>
            <button className="example-chip" onClick={() => window.dispatchEvent(new CustomEvent('sendMessage', { detail: 'How to make rendang?' }))}>
              How to make rendang?
            </button>
          </div>
        </div>
      )}
      {messages.map((msg, i) => (
        <MessageBubble key={i} message={msg} />
      ))}
      {isLoading && <TypingIndicator />}
      <div ref={bottomRef} />
    </div>
  )
}

function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('')
  const textareaRef = useRef(null)

  useEffect(() => {
    const handler = (e) => {
      setText(e.detail)
      if (textareaRef.current) {
        textareaRef.current.focus()
      }
    }
    window.addEventListener('sendMessage', handler)
    return () => window.removeEventListener('sendMessage', handler)
  }, [])

  const handleSubmit = () => {
    const trimmed = text.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setText('')
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const handleInput = (e) => {
    setText(e.target.value)
    e.target.style.height = 'auto'
    e.target.style.height = Math.min(e.target.scrollHeight, 150) + 'px'
  }

  return (
    <div className="chat-input-area">
      <div className="input-wrapper">
        <textarea
          ref={textareaRef}
          className="chat-input"
          placeholder="Tanyakan resep atau cara memasak..."
          value={text}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
        />
        <button className="send-button" onClick={handleSubmit} disabled={disabled || !text.trim()}>
          <FiSend className="icon-sm" />
        </button>
      </div>
    </div>
  )
}

export default function App() {
  const [messages, setMessages] = useState(loadMessages)
  const [isLoading, setIsLoading] = useState(false)
  const [sessionId] = useState(loadSessionId)
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [activeChatId, setActiveChatId] = useState(null)

  useEffect(() => {
    if (!activeChatId) return

    const activeChat = getChatById(activeChatId)
    if (activeChat) {
      setMessages(normalizeMessages(activeChat.messages))
    }
  }, [activeChatId])

  useEffect(() => {
    if (activeChatId) return
    saveMessages(messages)
  }, [messages, activeChatId])

  const handleSend = useCallback(async (text) => {
    const userMsg = { role: 'user', content: text }

    let chatId = activeChatId
    if (!chatId) {
      const newChat = createNewChat(text)
      chatId = newChat.id
      setActiveChatId(chatId)
      setSidebarOpen(false)
    }

    const history = normalizeMessages(messages)
    setMessages(prev => [...prev, userMsg])
    addMessageToChat(chatId, userMsg)
    setIsLoading(true)

    try {
      // Wake up the server (free tier may be sleeping)
      try {
        const healthCtrl = new AbortController()
        setTimeout(() => healthCtrl.abort(), 15000)
        await fetch(`${API_URL}/health`, { method: 'GET', signal: healthCtrl.signal })
      } catch {
        // Server might still be waking up, wait a bit
        await new Promise(r => setTimeout(r, 5000))
      }

      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), 120000)

      const chatUrl = `${API_URL}/chat`
      console.log('Fetching:', chatUrl)

      // Detect if user is ready for the recipe (short affirmative responses only)
      const trimmedText = text.toLowerCase().trim()
      const readySignals = ['sudah', 'langsung aja', 'cukup', 'skip', 'gas', 'ready', 'yup', 'siap', 'langsung']
      const isReady = (
        readySignals.includes(trimmedText) ||
        (trimmedText.length <= 10 && readySignals.some(s => trimmedText === s || trimmedText === s + '!' || trimmedText === s + '.'))
      )

      const res = await fetch(chatUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId, history, confirmed: isReady }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId)

      if (!res.ok) {
        const errText = await res.text()
        console.error('API error:', res.status, errText)
        throw new Error('API error ' + res.status + ' at ' + chatUrl + ': ' + errText)
      }

      const data = await res.json()
      if (!data.reply) {
        console.error('No reply in response:', data)
        throw new Error('No reply in response')
      }
      const botMsg = {
        role: 'assistant',
        content: data.reply,
        sources: data.sources || [],
      }
      setMessages(prev => [...prev, botMsg])
      addMessageToChat(chatId, botMsg)
    } catch (err) {
      console.error('Chat error:', err.message)
      const errorMsg = {
        role: 'assistant',
        content: `Maaf, terjadi kesalahan.\n\nError: ${err.message}`,
      }
      setMessages(prev => [...prev, errorMsg])
      if (chatId) {
        addMessageToChat(chatId, errorMsg)
      }
    } finally {
      setIsLoading(false)
    }
  }, [activeChatId, messages, sessionId])

  const handleSelectChat = useCallback((chat) => {
    setActiveChatId(chat?.id ?? null)
    setMessages(normalizeMessages(chat?.messages))
    setSidebarOpen(false)
  }, [])

  const handleNewChat = useCallback(() => {
    setActiveChatId(null)
    setMessages([])
    setSidebarOpen(false)
  }, [])

  return (
    <ThemeProvider>
      <div className="app-container">
        <div style={{ display: 'flex', minHeight: '100vh' }}>
          <ChatSidebar
            isOpen={sidebarOpen}
            onToggle={() => setSidebarOpen(!sidebarOpen)}
            onSelect={handleSelectChat}
            onNewChat={handleNewChat}
            activeChatId={activeChatId}
          />
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <Header />
            <MessageList messages={messages} isLoading={isLoading} />
            <ChatInput onSend={handleSend} disabled={isLoading} />
          </div>
        </div>
      </div>
    </ThemeProvider>
  )
}
