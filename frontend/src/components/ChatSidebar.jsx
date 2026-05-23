import React, { useState, useEffect, useCallback, useMemo } from 'react'
import { FiMenu, FiPlus, FiMessageSquare, FiTrash2, FiX } from 'react-icons/fi'

const STORAGE_KEY = 'resepai_chats'

function loadChats() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]')
  } catch {
    return []
  }
}

function formatTimestamp(isoString) {
  if (!isoString) return ''
  const date = new Date(isoString)
  const now = new Date()
  const diffMs = now - date
  const diffMins = Math.floor(diffMs / 60000)
  const diffHours = Math.floor(diffMs / 3600000)
  const diffDays = Math.floor(diffMs / 86400000)

  if (diffMins < 1) return 'Baru saja'
  if (diffMins < 60) return `${diffMins} menit lalu`
  if (diffHours < 24) return `${diffHours} jam lalu`
  if (diffDays < 7) return `${diffDays} hari lalu`

  return date.toLocaleDateString('id-ID', {
    day: 'numeric',
    month: 'short',
    year: date.getFullYear() !== now.getFullYear() ? 'numeric' : undefined,
  })
}

function truncate(str, maxLen = 40) {
  if (!str) return 'Percakapan baru'
  return str.length > maxLen ? str.slice(0, maxLen) + '…' : str
}

export default function ChatSidebar({ isOpen, onToggle, onSelect, onNewChat, activeChatId }) {
  const [chats, setChats] = useState([])

  const refreshChats = useCallback(() => {
    setChats(loadChats())
  }, [])

  useEffect(() => {
    refreshChats()
    const interval = setInterval(refreshChats, 2000)
    return () => clearInterval(interval)
  }, [refreshChats])

  useEffect(() => {
    if (isOpen) refreshChats()
  }, [isOpen, refreshChats])

  const handleDelete = (e, chatId) => {
    e.stopPropagation()
    const updated = loadChats().filter(c => c.id !== chatId)
    localStorage.setItem(STORAGE_KEY, JSON.stringify(updated))
    setChats(updated)
  }

  const sortedChats = useMemo(
    () => [...chats].sort(
      (a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt)
    ),
    [chats]
  )

  return (
    <>
      {/* Mobile overlay backdrop */}
      {isOpen && (
        <div
          className="fixed inset-0 z-30 bg-black/40 backdrop-blur-sm md:hidden"
          onClick={onToggle}
        />
      )}

      {/* Toggle button — visible only on mobile, positioned at far left */}
      <button
        onClick={onToggle}
        className="fixed top-4 left-4 z-40 inline-flex h-11 w-11 items-center justify-center rounded-full border transition hover:-translate-y-0.5 md:hidden"
        style={{
          background: 'var(--bg-elevated)',
          color: 'var(--text)',
          borderColor: 'var(--border)',
          boxShadow: '0 10px 24px rgba(15, 23, 42, 0.08)',
        }}
        aria-label={isOpen ? 'Tutup sidebar' : 'Buka sidebar'}
      >
        {isOpen ? <FiX className="h-5 w-5" /> : <FiMenu className="h-5 w-5" />}
      </button>

      {/* Sidebar panel */}
      <aside
        className={`
          fixed top-0 left-0 z-30 flex h-full w-72 flex-col border-r
          transition-transform duration-300 ease-in-out
          md:relative md:z-20
          ${isOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0 md:w-0 md:border-0 md:overflow-hidden'}
        `}
        style={{
          background: 'var(--bg-elevated)',
          borderColor: 'var(--border)',
        }}
      >
        {/* Header */}
        <div className="flex items-center justify-between px-4 py-4 border-b" style={{ borderColor: 'var(--border)' }}>
          <h2 className="text-sm font-semibold tracking-tight" style={{ color: 'var(--text)' }}>
            Riwayat Chat
          </h2>
          <button
            onClick={onNewChat}
            className="inline-flex h-9 w-9 items-center justify-center rounded-lg transition hover:-translate-y-0.5"
            style={{
              background: 'linear-gradient(135deg, var(--accent), var(--accent-hover))',
              color: 'var(--accent-foreground)',
              boxShadow: '0 8px 20px rgba(59, 130, 246, 0.22)',
            }}
            aria-label="Chat baru"
            title="Chat baru"
          >
            <FiPlus className="h-4 w-4" />
          </button>
        </div>

        {/* Chat list */}
        <div className="flex-1 overflow-y-auto px-2 py-3">
          {sortedChats.length === 0 && (
            <div className="flex flex-col items-center justify-center py-12 text-center px-4">
              <FiMessageSquare className="h-8 w-8 mb-3" style={{ color: 'var(--text-muted)', opacity: 0.5 }} />
              <p className="text-xs" style={{ color: 'var(--text-muted)' }}>
                Belum ada percakapan
              </p>
              <p className="text-xs mt-1" style={{ color: 'var(--text-muted)', opacity: 0.7 }}>
                Mulai chat baru untuk menyimpan riwayat
              </p>
            </div>
          )}

          {sortedChats.length > 0 && (
            <ul role="list" className="space-y-1">
              {sortedChats.map(chat => {
                const isActive = chat.id === activeChatId
                return (
                  <li key={chat.id}>
                    <button
                      onClick={() => onSelect(chat)}
                      className={`
                        group w-full flex items-start gap-3 rounded-xl px-3 py-3 text-left transition-all duration-150
                        ${isActive ? 'ring-1' : 'hover:translate-x-0.5'}
                      `}
                      style={{
                        background: isActive
                          ? 'color-mix(in srgb, var(--accent) 12%, var(--bg-elevated))'
                          : 'transparent',
                        '--tw-ring-color': isActive ? 'var(--accent)' : undefined,
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.background = 'color-mix(in srgb, var(--accent) 6%, var(--bg-elevated))'
                        }
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) {
                          e.currentTarget.style.background = 'transparent'
                        }
                      }}
                    >
                      <div
                        className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg mt-0.5"
                        style={{
                          background: isActive
                            ? 'linear-gradient(135deg, var(--accent), var(--accent-hover))'
                            : 'var(--bg-muted)',
                          color: isActive ? 'var(--accent-foreground)' : 'var(--text-muted)',
                        }}
                      >
                        <FiMessageSquare className="h-4 w-4" />
                      </div>

                      <div className="flex-1 min-w-0">
                        <p
                          className="text-sm font-medium truncate leading-5"
                          style={{ color: 'var(--text)' }}
                        >
                          {truncate(chat.title)}
                        </p>
                        <p
                          className="text-xs mt-0.5"
                          style={{ color: 'var(--text-muted)' }}
                        >
                          {formatTimestamp(chat.updatedAt || chat.createdAt)}
                        </p>
                      </div>

                      <button
                        onClick={(e) => handleDelete(e, chat.id)}
                        className="shrink-0 mt-1 inline-flex h-7 w-7 items-center justify-center rounded-lg opacity-0 group-hover:opacity-100 group-focus:opacity-100 focus:opacity-100 focus-visible:opacity-100 transition-opacity"
                        style={{ color: 'var(--text-muted)' }}
                        onMouseEnter={(e) => {
                          e.currentTarget.style.color = '#ef4444'
                          e.currentTarget.style.background = 'rgba(239, 68, 68, 0.1)'
                        }}
                        onMouseLeave={(e) => {
                          e.currentTarget.style.color = 'var(--text-muted)'
                          e.currentTarget.style.background = 'transparent'
                        }}
                        aria-label="Hapus percakapan"
                        title="Hapus percakapan"
                      >
                        <FiTrash2 className="h-3.5 w-3.5" />
                      </button>
                    </button>
                  </li>
                )
              })}
            </ul>
          )}
        </div>

        {/* Footer */}
        <div className="px-4 py-3 border-t" style={{ borderColor: 'var(--border)' }}>
          <p className="text-xs text-center" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>
            ResepAI Chat History
          </p>
        </div>
      </aside>
    </>
  )
}
