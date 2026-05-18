# Backend Implementation Brief — ResepAI Recipe Chatbot

## Goal
Create an Express.js backend with LangChain RAG pipeline + ChromaDB for an Indonesian recipe chatbot.

## Tech Stack
- **Runtime:** Node.js + Express.js
- **LLM:** `google/gemini-2.0-flash-free` via OpenRouter (API key in env)
- **Embeddings:** Use free local model `all-MiniLM-L6-v2` via `@xenova/transformers` (no cost)
- **Vector DB:** ChromaDB (local, embedded mode via `chromadb` npm package)
- **LangChain:** `@langchain/core`, `@langchain/community`, `@langchain/chroma`
- **Dataset:** `junwatu/indonesian-recipes` from HuggingFace (Parquet format)

## Project Structure
```
backend/
├── package.json
├── .env
├── src/
│   ├── server.js          # Express server entry point
│   ├── ingest.js          # Dataset ingestion + ChromaDB embedding script
│   ├── rag.js             # RAG pipeline (retrieval + generation)
│   └── routes/
│       └── chat.js        # POST /api/chat endpoint
```

## Implementation Steps

### 1. Initialize project
- `npm init -y` in `/home/ubuntu/recipe-chatbot/backend/`
- Install dependencies: express, cors, dotenv, chromadb, @langchain/core, @langchain/community, @langchain/chroma, @xenova/transformers, node-fetch
- Create `.env` with: `OPENROUTER_API_KEY`, `PORT=3001`

### 2. Write `src/ingest.js` — Dataset Ingestion
- Download `junwatu/indonesian-recipes` dataset from HuggingFace
- Use `parquet-wasm` or fetch JSON conversion — simplest: use HuggingFace's `https://datasets-server.huggingface.co/rows?dataset=junwatu%2Findonesian-recipes&config=default&split=train&offset=0&length=100` API to fetch rows in JSON
- For each recipe row, create a document with:
  - `pageContent`: formatted string with title, ingredients (joined), steps (joined)
  - `metadata`: { title, num_ingredients, num_steps, source: "indonesian-recipes" }
- Embed using Xenova's `all-MiniLM-L6-v2` (runs locally, free)
- Store in ChromaDB collection named `recipes`
- Log progress (e.g., "Ingested 100/5000 recipes")
- Make it idempotent: skip if collection already has documents

### 3. Write `src/rag.js` — RAG Pipeline
- Initialize ChromaDB client, connect to `recipes` collection
- Create embedding function using Xenova `all-MiniLM-L6-v2`
- Retrieval: similarity search with topK=5
- If top similarity score < 0.5, flag as low-confidence (LLM should note it's not from database)
- Build system prompt: "You are ResepAI, a helpful Indonesian recipe assistant. Answer in the same language as the user (Bahasa Indonesia or English). Use the provided recipe context to answer. If context is low confidence, say you're providing general cooking advice, not from the database."
- Call OpenRouter API (`https://openrouter.ai/api/v1/chat/completions`) with:
  - model: `google/gemini-2.0-flash-free`
  - messages: system + conversation history + user query
  - temperature: 0.7
  - max_tokens: 1024
- Return: { reply, sources: [{ title, num_ingredients, num_steps }] }

### 4. Write `src/routes/chat.js` — Chat Endpoint
- POST `/api/chat`
- Body: { message: string, history: [{role, content}] (optional) }
- Validate: message must be non-empty string, max 500 chars
- Limit history to last 10 exchanges
- Call RAG pipeline
- Return: { reply, sources }
- Error handling: return 400 for bad input, 500 for server errors with friendly message

### 5. Write `src/server.js` — Express Server
- Express + CORS (allow all origins for now)
- Parse JSON bodies
- Mount `/api/chat` route
- Health check: GET `/api/health` → { status: "ok", timestamp }
- Start server on PORT from env (default 3001)
- After server starts, run ingestion if collection is empty

## Important Notes
- All code in `/home/ubuntu/recipe-chatbot/backend/`
- Use CommonJS (`require`) or ES modules consistently — pick one
- Add error handling everywhere
- Log meaningful messages (ingestion progress, request/response info)
- The frontend is at `/home/ubuntu/recipe-chatbot/frontend/` — don't modify it
- After all code is written, run `npm install` and test that the server starts
- Test ingestion with a small batch first (limit=10), then full dataset
- Commit all files to git when done
