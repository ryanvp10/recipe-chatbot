const express = require('express');

const { generateRecipeReply } = require('../rag');

const router = express.Router();

function sanitizeHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .filter((item) => item && (item.role === 'user' || item.role === 'assistant') && typeof item.content === 'string')
    .slice(-20);
}

router.post('/', async (req, res) => {
  const { message, history } = req.body || {};

  if (typeof message !== 'string' || !message.trim()) {
    return res.status(400).json({ error: 'Message must be a non-empty string.' });
  }

  if (message.trim().length > 500) {
    return res.status(400).json({ error: 'Message must be 500 characters or fewer.' });
  }

  try {
    const safeHistory = sanitizeHistory(history);
    console.log('[chat] Processing request. messageLength=%d historyMessages=%d', message.trim().length, safeHistory.length);
    const result = await generateRecipeReply(message.trim(), safeHistory);
    return res.json({ reply: result.reply, sources: result.sources });
  } catch (error) {
    console.error('[chat] Failed to generate chat response:', error);
    return res.status(500).json({ error: 'Sorry, something went wrong while generating the recipe response.' });
  }
});

module.exports = router;
