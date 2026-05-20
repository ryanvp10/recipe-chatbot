const STORAGE_KEY = 'resepai_chats';

function generateId() {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function loadAll() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEY) || '[]');
  } catch {
    return [];
  }
}

function saveAll(chats) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(chats));
  } catch (e) {
    console.error('[chatStorage] Failed to save:', e);
  }
}

export function getAllChats() {
  const chats = loadAll();
  return chats.sort(
    (a, b) => new Date(b.updatedAt || b.createdAt) - new Date(a.updatedAt || a.createdAt)
  );
}

export function getChatById(id) {
  const chats = loadAll();
  return chats.find(c => c.id === id) || null;
}

export function saveChat(chat) {
  const chats = loadAll();
  const idx = chats.findIndex(c => c.id === chat.id);
  if (idx >= 0) {
    chats[idx] = { ...chats[idx], ...chat, updatedAt: new Date().toISOString() };
  } else {
    chats.push(chat);
  }
  saveAll(chats);
}

export function deleteChat(id) {
  const chats = loadAll().filter(c => c.id !== id);
  saveAll(chats);
}

export function createNewChat(firstMessage) {
  const now = new Date().toISOString();
  const chat = {
    id: generateId(),
    title: firstMessage ? firstMessage.slice(0, 40) : 'Percakapan baru',
    messages: [],
    createdAt: now,
    updatedAt: now,
  };
  saveChat(chat);
  return chat;
}

export function addMessageToChat(chatId, message) {
  const chats = loadAll();
  const idx = chats.findIndex(c => c.id === chatId);
  if (idx < 0) return null;

  chats[idx].messages.push(message);
  chats[idx].updatedAt = new Date().toISOString();

  // If first user message, update title
  if (message.role === 'user' && chats[idx].messages.filter(m => m.role === 'user').length === 1) {
    chats[idx].title = message.content.slice(0, 40);
  }

  saveAll(chats);
  return chats[idx];
}
