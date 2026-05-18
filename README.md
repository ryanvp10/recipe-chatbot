# ResepAI — Indonesian Recipe Chatbot

An AI-powered conversational chatbot for Indonesian recipes and cooking assistance, built with LangChain RAG, ChromaDB, and React.

## Features

- 🍳 **Recipe Search** — Find Indonesian recipes by ingredient, dish name, or cooking method
- 💬 **Conversational** — Multi-turn cooking assistant with memory
- 🧠 **RAG-Powered** — Retrieves from 10K+ Indonesian recipes dataset
- 🌐 **Bilingual** — Responds in Bahasa Indonesia or English (auto-detect)
- 🎨 **Light/Dark Mode** — Toggle between themes
- 📱 **Responsive** — Works on desktop and mobile
- 🔓 **Fully Public** — No login required

## Tech Stack

| Layer | Technology |
|-------|------------|
| Frontend | React + Vite |
| Backend | Express.js (Node.js) |
| RAG | LangChain.js |
| Vector DB | ChromaDB (embedded) |
| LLM | Google Gemini 2.0 Flash (via OpenRouter) |
| Dataset | [junwatu/indonesian-recipes](https://huggingface.co/datasets/junwatu/indonesian-recipes) |

## Project Structure

```
recipe-chatbot/
├── backend/
│   ├── package.json
│   ├── .env
│   ├── server.js                  # Express server entry
│   ├── config/
│   │   └── index.js               # Config loader
│   ├── scripts/
│   │   └── ingest.js              # Dataset → ChromaDB ingestion
│   ├── routes/
│   │   └── chat.js                # POST /api/chat
│   ├── services/
│   │   ├── rag.js                 # RAG pipeline
│   │   ├── memory.js              # Conversation memory
│   │   └── chroma.js              # ChromaDB client
│   └── data/
│       └── chroma/                # ChromaDB storage (gitignored)
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── context/
│       │   └── ThemeContext.jsx
│       ├── components/
│       │   ├── ChatWindow.jsx
│       │   ├── MessageList.jsx
│       │   ├── MessageBubble.jsx
│       │   ├── ChatInput.jsx
│       │   ├── Header.jsx
│       │   └── TypingIndicator.jsx
│       ├── hooks/
│       │   └── useChat.js
│       └── styles/
│           └── global.css
└── README.md
```

## Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- OpenRouter API key

### Backend Setup

```bash
cd backend
npm install
cp .env.example .env
# Edit .env with your OpenRouter API key
node scripts/ingest.js   # One-time: load dataset into ChromaDB
npm run dev              # Start server on port 3001
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev              # Start dev server on port 5173
```

### Environment Variables

`backend/.env`:
```
OPENROUTER_API_KEY=sk-or-v1-...
CHROMA_DB_PATH=./data/chroma
PORT=3001
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/chat` | Send message, get recipe response |
| GET | `/api/health` | Health check + recipe count |

### POST /api/chat

**Request:**
```json
{
  "message": "Resep ayam goreng?",
  "sessionId": "abc123",
  "history": []
}
```

**Response:**
```json
{
  "reply": "Berikut resep ayam goreng...",
  "sources": [
    { "title": "Ayam Goreng Kuning", "score": 0.92 }
  ],
  "sessionId": "abc123"
}
```

## Architecture

```
[React UI] ←→ [Express API] ←→ [LangChain RAG]
                                   ↕
                               [ChromaDB]
                                   ↕
                            [OpenRouter LLM]
```

1. User sends message
2. Backend embeds query → searches ChromaDB (top 5 recipes)
3. If recipes found: inject as context → generate response
4. If no recipes: LLM answers with strict cooking-only fallback
5. Conversation memory maintained per session

## License

MIT
