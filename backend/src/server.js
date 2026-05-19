const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const rateLimit = require('express-rate-limit');

dotenv.config();

const { generateRecipeReply, embeddingsReady } = require('./rag');

const app = express();
const port = Number.parseInt(process.env.PORT, 10) || 3001;

const allowedOrigins = (
  process.env.ALLOWED_ORIGINS || 'http://localhost:5173,http://localhost:3000,https://recipe-chat.netlify.app'
)
  .split(',')
  .map((origin) => origin.trim())
  .filter(Boolean);

app.use(
  cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type'],
    credentials: false,
  })
);

// Handle OPTIONS preflight explicitly for HF Space proxy compatibility
app.options('*', (req, res) => {
  const origin = req.headers.origin;
  if (allowedOrigins.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, OPTIONS');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
    res.setHeader('Access-Control-Max-Age', '86400');
  }
  res.status(204).send();
});

app.use(express.json({ limit: '1mb' }));

// Trust proxy for correct client IP behind reverse proxies (Railway, etc.)
app.set('trust proxy', 1);

const chatLimiter = rateLimit({
  windowMs: 15 * 60 * 1000,
  max: 100,
  standardHeaders: true,
  legacyHeaders: false,
  message: { error: 'Too many requests, please try again later.' },
});

app.get('/api/health', (_req, res) => {
  res.json({
    status: 'ok',
    timestamp: new Date().toISOString(),
    embeddingsReady: embeddingsReady(),
  });
});

app.post('/api/chat', chatLimiter, async (req, res) => {
  try {
    const { message, history, confirmed } = req.body;

    if (!message || typeof message !== 'string' || !message.trim()) {
      return res.status(400).json({ error: 'Message is required and must be a non-empty string.' });
    }

    if (message.length > 2000) {
      return res.status(400).json({ error: 'Message too long. Maximum 2000 characters.' });
    }

    if (history !== undefined && !Array.isArray(history)) {
      return res.status(400).json({ error: 'History must be an array if provided.' });
    }

    if (history && history.length > 20) {
      return res.status(400).json({ error: 'History too long. Maximum 20 messages.' });
    }

    if (!embeddingsReady()) {
      return res.status(503).json({ error: 'Service initializing, please try again in a moment.' });
    }

    const result = await generateRecipeReply(message.trim(), history, confirmed === true);
    res.json(result);
  } catch (error) {
    console.error('[server] Chat error:', error);
    res.status(500).json({ error: 'Failed to generate response.' });
  }
});

app.use((err, _req, res, _next) => {
  console.error('[server] Unhandled Express error:', err);
  if (!res.headersSent) {
    res.status(500).json({ error: 'Internal server error.' });
  }
});

const server = app.listen(port, () => {
  console.log(`[server] ResepAI backend listening on port ${port}`);
  console.log(`[server] CORS allowed origins: ${allowedOrigins.join(', ')}`);
});

server.on('error', (error) => {
  console.error('[server] Failed to start server:', error);
  process.exit(1);
});
