import React, { useState, useEffect, useRef, useCallback, createContext, useContext } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeSanitize from 'rehype-sanitize'
import { FiMoon, FiSend, FiSun } from 'react-icons/fi'
import { GiChefToque } from 'react-icons/gi'

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
        {message.sources && message.sources.length > 0 && (
          <div className="message-sources">
            <span className="sources-label">📚 Sumber:</span>
            {message.sources.map((s, i) => (
              <span key={i} className="source-item">{s.title}</span>
            ))}
          </div>
        )}
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

  useEffect(() => {
    saveMessages(messages)
  }, [messages])

  const handleSend = useCallback(async (text) => {
    const userMsg = { role: 'user', content: text }
    setMessages(prev => [...prev, userMsg])
    setIsLoading(true)

    try {
      const history = messages
        .filter(m => m.role !== 'user' || messages.indexOf(m) < messages.length)
        .map(m => ({ role: m.role, content: m.content }))

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 60000);

      const res = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: text, sessionId, history }),
        signal: controller.signal,
      })

      clearTimeout(timeoutId);

      if (!res.ok) {
        const errText = await res.text();
        console.error('API error:', res.status, errText);
        throw new Error('API error ' + res.status + ': ' + errText);
      }

      const data = await res.json()
      if (!data.reply) {
        console.error('No reply in response:', data);
        throw new Error('No reply in response');
      }
      const botMsg = {
        role: 'assistant',
        content: data.reply,
        sources: data.sources || [],
      }
      setMessages(prev => [...prev, botMsg])
    } catch (err) {
      console.error('Chat error:', err.message);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Maaf, terjadi kesalahan. Silakan coba lagi.',
      }])
    } finally {
      setIsLoading(false)
    }
  }, [messages, sessionId])

  return (
    <ThemeProvider>
      <div className="app-container">
        <Header />
        <MessageList messages={messages} isLoading={isLoading} />
        <ChatInput onSend={handleSend} disabled={isLoading} />
      </div>
    </ThemeProvider>
  )
}
