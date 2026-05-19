const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const LLM_URL = 'https://api.freemodel.dev/v1/chat/completions';
const HF_EMBEDDING_URL = 'https://router.huggingface.co/hf-inference/v1/pipeline/feature-extraction/BAAI/bge-small-en-v1.5';
const MODEL_NAME = 'gpt-5.4';
const SYSTEM_PROMPT =
  'You are ResepAI, a friendly Indonesian cooking buddy. Match the user\'s language (Bahasa Indonesia or English). Stay strictly within cooking and food topics only. If the user asks about anything outside cooking or food, politely redirect with: "Maaf, saya hanya bisa membantu soal masak-masak dan resep. Ada yang bisa dibantu soal makanan? 😊"\n\n## CRITICAL RULES:\n1. If the "RECIPE CONTEXT" section below is EMPTY or says "NO RECIPE CONTEXT PROVIDED", you are in DISCUSSION MODE. In discussion mode:\n   - Do NOT give any recipe, ingredients, or cooking steps\n   - ONLY ask one clarifying question at a time to help narrow down what the user wants\n   - Question order: variant/type → available ingredients → portion size → cooking time → other preferences\n   - Keep the conversation fun and casual, share food origins/fun facts when relevant\n   - Examples of food knowledge: Nasi liwet from Solo, Rendang from West Sumatra, Sate Madura from Madura island, etc.\n\n2. If the "RECIPE CONTEXT" section below has actual recipe content, you are in RECIPE MODE. In recipe mode:\n   - Give the full detailed recipe using the provided context\n   - Include ingredients and step-by-step instructions\n   - Be warm and helpful\n\n3. Be warm, casual, helpful, use "kamu", and occasional emoji is okay.\n4. If context is low confidence, say you are providing general cooking advice, not from the database.';
const MAX_HISTORY_MESSAGES = 20;
const MAX_MESSAGE_LENGTH = 2000;
const MAX_CONTEXT_LENGTH = 8000;
const FETCH_TIMEOUT_MS = 120000;

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
      content: item.content.trim().slice(0, MAX_MESSAGE_LENGTH),
    }));
}

async function embedQuery(text) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(HF_EMBEDDING_URL, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${process.env.HF_TOKEN}`,
      },
      body: JSON.stringify({ inputs: text.slice(0, 500) }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      throw new Error('HF embedding request timed out');
    }
    throw err;
  }
  clearTimeout(timeout);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`HF embedding request failed with status ${response.status}: ${errorText}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error(`Failed to parse HF embedding response: ${err.message}`);
  }

  // BGE-small via HF Inference returns a flat array of floats
  const vector = Array.isArray(data) ? data : null;
  if (!vector || vector.length === 0) {
    throw new Error('HF embedding response did not include a valid embedding vector.');
  }

  return vector;
}

async function retrieveContext(query) {
  let queryEmbedding;
  try {
    queryEmbedding = await embedQuery(query);
  } catch (err) {
    console.error('[rag] Embedding failed:', err.message);
    return { context: '', sources: [], lowConfidence: true, embeddingError: true };
  }

  let results;
  try {
    results = search(queryEmbedding, 5);
  } catch (err) {
    console.error('[rag] Search failed:', err.message);
    return { context: '', sources: [], lowConfidence: true, searchError: true };
  }

  const contextBlocks = [];
  const sources = [];
  let lowConfidence = true;

  for (const r of results) {
    if (r.similarity >= 0.15) {
      lowConfidence = false;
    }
    contextBlocks.push(`Similarity: ${r.similarity.toFixed(2)}\n${r.document}`);
    sources.push({
      title: r.metadata?.title || 'Unknown recipe',
      num_ingredients: r.metadata?.num_ingredients || 0,
      num_steps: r.metadata?.num_steps || 0,
    });
  }

  let context = contextBlocks.join('\n\n---\n\n');
  // Truncate context if too long
  if (context.length > MAX_CONTEXT_LENGTH) {
    context = context.slice(0, MAX_CONTEXT_LENGTH) + '\n... [truncated]';
  }

  return { context, sources, lowConfidence };
}

async function callLLM(messages) {
  const apiKey = process.env.FREEMODEL_API_KEY;
  if (!apiKey) {
    throw new Error('FREEMODEL_API_KEY is not configured.');
  }

  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  let response;
  try {
    response = await fetch(LLM_URL, {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${apiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages,
        temperature: 0.7,
        max_tokens: 1024,
      }),
      signal: controller.signal,
    });
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      throw new Error('LLM request timed out');
    }
    throw err;
  }
  clearTimeout(timeout);

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`LLM request failed with status ${response.status}: ${errorText}`);
  }

  let data;
  try {
    data = await response.json();
  } catch (err) {
    throw new Error(`Failed to parse LLM response: ${err.message}`);
  }

  const reply = data?.choices?.[0]?.message?.content;
  if (!reply || typeof reply !== 'string') {
    throw new Error('LLM response did not include a valid reply.');
  }

  return reply;
}

async function generateRecipeReply(message, history = [], confirmed = false) {
  const safeHistory = sanitizeHistory(history);
  const retrieval = confirmed
    ? await retrieveContext(message)
    : { context: '', sources: [], lowConfidence: true };
  const { context, sources, lowConfidence } = retrieval;

  const systemContent = [
    SYSTEM_PROMPT,
    confirmed
      ? lowConfidence
        ? 'Retrieved context confidence is low. Explicitly say when advice is general and not directly from the recipe database.'
        : 'Retrieved context is considered relevant. Prefer the recipe database details when answering.'
      : 'You are in DISCUSSION MODE. Do NOT give any recipe, ingredients, or cooking steps. ONLY ask one clarifying question.',
    '--- RECIPE CONTEXT START ---',
    confirmed ? (context || 'No recipe context available.') : 'NO RECIPE CONTEXT PROVIDED - You are in discussion mode. Only ask questions.',
    '--- RECIPE CONTEXT END ---',
  ].join('\n\n');

  const messages = [
    { role: 'system', content: systemContent },
    ...safeHistory,
    { role: 'user', content: message.slice(0, MAX_MESSAGE_LENGTH) },
  ];

  log('Generating reply with', sources.length, 'sources. Low confidence:', lowConfidence);
  const reply = await callLLM(messages);
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
