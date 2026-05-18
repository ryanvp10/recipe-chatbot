const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');

const chatRouter = require('./routes/chat');

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3001;

app.use(cors());
app.use(express.json({ limit: '1mb' }));

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/chat', chatRouter);

app.use((err, _req, res, _next) => {
  console.error('[server] Unhandled Express error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

const server = app.listen(port, async () => {
  console.log(`[server] ResepAI backend listening on port ${port}`);
  console.log('[server] Note: Make sure ChromaDB is running on localhost:8000');
  console.log('[server] Start ChromaDB with: chromadb run --path ./chroma_data');
});

server.on('error', (error) => {
  console.error('[server] Failed to start server:', error);
  process.exit(1);
});
