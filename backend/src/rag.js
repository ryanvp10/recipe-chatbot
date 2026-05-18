const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { ChromaClient } = require('chromadb');
const { embedTexts, COLLECTION_NAME, CHROMA_URL } = require('./ingest');

dotenv.config();

const OPENROUTER_URL = 'https://openrouter.ai/api/v1/chat/completions';
const MODEL_NAME = 'google/gemini-2.0-flash-free';
const SYSTEM_PROMPT = 'You are ResepAI, a helpful Indonesian recipe assistant. Answer in the same language as the user (Bahasa Indonesia or English). Use the provided recipe context to answer. If context is low confidence, say you are providing general cooking advice, not from the database.';
const MAX_HISTORY_MESSAGES = 20;

let collectionPromise;

function log(...args) {
  console.log('[rag]', ...args);
}

function sanitizeHistory(history) {
  if (!Array.isArray(history)) return [];

  return history
    .filter((item) => item && (item.role === 'user' || item.role === 'assistant') && typeof item.content === 'string')
    .slice(-MAX_HISTORY_MESSAGES)
    .map((item) => ({ role: item.role, content: item.content.trim().slice(0, 2000) }));
}

async function getCollection() {
  if (!collectionPromise) {
    const client = new ChromaClient({ url: CHROMA_URL });
    collectionPromise = client.getOrCreateCollection({ name: COLLECTION_NAME });
  }
  return collectionPromise;
}

async function retrieveContext(query) {
  const collection = await getCollection();
  const queryEmbedding = await embedTexts([query]).then(([emb]) => emb);

  const results = await collection.query({
    queryEmbeddings: [queryEmbedding],
    nResults: 5,
    include: ['documents', 'metadatas', 'distances'],
  });

  const sources = [];
  const contextBlocks = [];
  let lowConfidence = true;

  const docs = results.documents?.[0] || [];
  const metas = results.metadatas?.[0] || [];
  const dists = results.distances?.[0] || [];

  for (let i = 0; i < docs.length; i++) {
    const distance = Number.isFinite(dists[i]) ? dists[i] : 1;
    // ChromaDB uses distance (lower = more similar), convert to similarity
    const similarity = 1 - distance;
    if (similarity >= 0.5) {
      lowConfidence = false;
    }

    contextBlocks.push(`Similarity: ${similarity.toFixed(2)}\n${docs[i]}`);
    sources.push({
      title: metas[i]?.title || 'Unknown recipe',
      num_ingredients: metas[i]?.num_ingredients || 0,
      num_steps: metas[i]?.num_steps || 0,
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

module.exports = {
  generateRecipeReply,
};
