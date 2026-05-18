const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const rateLimit = require('express-rate-limit');

const chatRouter = require('./routes/chat');

dotenv.config();

const app = express();
const port = Number(process.env.PORT) || 3001;

const allowedOrigins = (process.env.ALLOWED_ORIGINS || 'http://localhost:5173,http://localhost:3000')
  .split(',')
  .map((origin) => origin.trim());

app.use(cors({
  origin: allowedOrigins,
  methods: ['GET', 'POST'],
  allowedHeaders: ['Content-Type'],
}));

app.use(express.json({ limit: '1mb' }));

const chatLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' },
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.use('/api/chat', chatLimiter, chatRouter);

app.use((err, _req, res, _next) => {
  console.error('[server] Unhandled Express error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

const server = app.listen(port, async () => {
  console.log(`[server] ResepAI backend listening on port ${port}`);
  console.log(`[server] CORS allowed origins: ${allowedOrigins.join(', ')}`);
  console.log('[server] Note: Make sure ChromaDB is running on localhost:8000');
});

server.on('error', (error) => {
  console.error('[server] Failed to start server:', error);
  process.exit(1);
});
