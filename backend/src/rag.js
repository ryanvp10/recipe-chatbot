const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { pipeline } = require('@xenova/transformers');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MODEL_NAME = 'google/gemini-2.0-flash-free';
const SYSTEM_PROMPT =
  'You are ResepAI, a helpful Indonesian recipe assistant. Answer in the same language as the user (Bahasa Indonesia or English). Use the provided recipe context to answer. If context is low confidence, say you are providing general cooking advice, not from the database.';
const MAX_HISTORY_MESSAGES = 20;

let extractorPromise;

function log(...args) {
  console.log('[rag]', ...args);
}

function sanitizeHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .filter(
      (item) =>
        item &&
        (item.role === 'user' || item.role === 'assistant') &&
        typeof item.content === 'string'
    )
    .slice(-MAX_HISTORY_MESSAGES)
    .map((item) => ({
      role: item.role,
      content: item.content.trim().slice(0, 2000),
    }));
}

async function getEmbeddingPipeline() {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractorPromise;
}

async function embedQuery(text) {
  const extractor = await getEmbeddingPipeline();
  const output = await extractor(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function retrieveContext(query) {
  const queryEmbedding = await embedQuery(query);
  const results = search(queryEmbedding, 5);

  const contextBlocks = [];
  const sources = [];
  let lowConfidence = true;

  for (const r of results) {
    if (r.similarity >= 0.5) {
      lowConfidence = false;
    }
    contextBlocks.push(`Similarity: ${r.similarity.toFixed(2)}\n${r.document}`);
    sources.push({
      title: r.metadata?.title || 'Unknown recipe',
      num_ingredients: r.metadata?.num_ingredients || 0,
      num_steps: r.metadata?.num_steps || 0,
    });
  }

  return {
    context: contextBlocks.join('\n\n---\n\n'),
    sources,
    lowConfidence,
  };
}

async function callOpenRouter(messages) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY is not configured.');
  }

  const response = await fetch(OPENROUTER_URL, {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
      'HTTP-Referer': 'http://localhost:3001',
      'X-Title': 'ResepAI Backend',
    },
    body: JSON.stringify({
      model: MODEL_NAME,
      messages,
      temperature: 0.7,
      max_tokens: 1024,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`OpenRouter request failed with status ${response.status}: ${errorText}`);
  }

  const data = await response.json();
  const reply = data?.choices?.[0]?.message?.content;
  if (!reply) {
    throw new Error('OpenRouter response did not include a reply.');
  }

  return reply;
}

async function generateRecipeReply(message, history = []) {
  const safeHistory = sanitizeHistory(history);
  const { context, sources, lowConfidence } = await retrieveContext(message);

  const systemContent = [
    SYSTEM_PROMPT,
    lowConfidence
      ? 'Retrieved context confidence is low. Explicitly say when advice is general and not directly from the recipe database.'
      : 'Retrieved context is considered relevant. Prefer the recipe database details when answering.',
    `Recipe context:\n${context || 'No recipe context available.'}`,
  ].join('\n\n');

  const messages = [
    { role: 'system', content: systemContent },
    ...safeHistory,
    { role: 'user', content: message },
  ];

  log('Generating reply with', sources.length, 'sources. Low confidence:', lowConfidence);
  const reply = await callOpenRouter(messages);
  return { reply, sources, lowConfidence };
}

// Load embeddings on module load
let embeddingsReady = false;
loadEmbeddings()
  .then(() => {
    embeddingsReady = true;
    log(`Embeddings ready. ${getEmbeddingCount()} recipes loaded.`);
  })
  .catch((err) => {
    console.error('[rag] Failed to load embeddings:', err);
  });

module.exports = {
  generateRecipeReply,
  embeddingsReady: () => embeddingsReady,
};
