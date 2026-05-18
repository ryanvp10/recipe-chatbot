const express = require('express');
const cors = require('cors');
const dotenv = require('dotenv');
const rateLimit = require('express-rate-limit');

dotenv.config();

const { generateRecipeReply, embeddingsReady } = require('./rag');

const app = express();
const port = Number(process.env.PORT) || 3001;

const allowedOrigins = (
  process.env.ALLOWED_ORIGINS || 'http://localhost:5173,http://localhost:3000'
)
  .split(',')
  .map((origin) => origin.trim());

app.use(
  cors({
    origin: allowedOrigins,
    methods: ['GET', 'POST'],
    allowedHeaders: ['Content-Type'],
  })
);

app.use(express.json({ limit: '1mb' }));

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
    const { message, history } = req.body;

    if (!message || typeof message !== 'string') {
      return res.status(400).json({ error: 'Message is required.' });
    }

    if (!embeddingsReady()) {
      return res.status(503).json({ error: 'Service initializing, please try again in a moment.' });
    }

    const result = await generateRecipeReply(message, history);
    res.json(result);
  } catch (error) {
    console.error('[server] Chat error:', error);
    res.status(500).json({ error: 'Failed to generate response.' });
  }
});

app.use((err, _req, res, _next) => {
  console.error('[server] Unhandled Express error:', err);
  res.status(500).json({ error: 'Internal server error.' });
});

const server = app.listen(port, () => {
  console.log(`[server] ResepAI backend listening on port ${port}`);
  console.log(`[server] CORS allowed origins: ${allowedOrigins.join(', ')}`);
});

server.on('error', (error) => {
  console.error('[server] Failed to start server:', error);
  process.exit(1);
});
