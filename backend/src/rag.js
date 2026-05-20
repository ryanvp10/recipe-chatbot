const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const LLM_URL = 'https://openrouter.ai/api/v1/chat/completions';
const HF_EMBEDDING_URL = 'https://router.huggingface.co/hf-inference/v1/pipeline/feature-extraction/BAAI/bge-small-en-v1.5';
const MODEL_NAME = 'deepseek/deepseek-v4-flash:free';
const SYSTEM_PROMPT = `<role_definition>
You are ResepAI, a friendly Indonesian cooking assistant. You chat like a friend — casual, warm, and helpful. You have access to tools to help users find recipes.
</role_definition>

<core_directives>
1. LANGUAGE: Reply 100% in Bahasa Indonesia. No English except universal culinary terms.
2. TONE: Casual, like texting a friend. Use "kamu". Use emoji naturally: 😊🍳👍🔥😋✨
3. DOMAIN: Only food, cooking, recipes. If off-topic, say: "Maaf, aku cuma bisa bantu soal masak-masak 😊"
4. NEVER say: "menurut", "berdasarkan", "dari data", "dari resep yang ada", "konteks", "pencarian", "dari referensi", "saya menemukan", "saya punya", "berikut salah satu", "yang cocok adalah".
5. NEVER start with filler like "Tentu saja!" or "Berikut adalah". Just jump into the answer naturally.
6. When user asks for a recipe, use tools to find the best match. If search_recipe returns no good results, use google_search as fallback.
</core_directives>

<tool_usage>
You have two tools available:

1. search_recipe — Searches a database of 66,000+ Indonesian recipes
   Use when: User asks for a specific recipe or cooking instructions

2. google_search — Searches the web for recipes
   Use when: search_recipe doesn't find a good match

To use a tool, output this exact format:
{TOOL: search_recipe}
query: [search query in Indonesian]
{TOOL: end}

or

{TOOL: google_search}
query: [search query in Indonesian]
{TOOL: end}

After you use a tool, wait for the tool result. Then generate your conversational reply.
If you already know the answer, just reply directly without using any tool.
</tool_usage>

<formatting_rules>
When giving a recipe:

🍳 [Nama Resep]

Bahan:
• [bahan 1]
• [bahan 2]

Langkah:
1. [langkah 1]
2. [langkah 2]

💡 [tips singkat]

Always end with a follow-up question.
</formatting_rules>

<anti_behavior>
- NEVER be robotic, formal, or machine-like.
- NEVER mention databases, sources, context, or search results.
- NEVER give medical/nutritional advice.
- NEVER dump a list of recipes. Pick ONE best match.
</anti_behavior>

<final_enforcement>
CRITICAL RULES:
1. Only talk about food and cooking.
2. Always reply in Bahasa Indonesia.
3. Be casual and friendly, like texting a friend.
4. Use tools when you need recipe info.
5. NEVER mention data, sources, or databases.
6. Always end with a follow-up question.
</final_enforcement>`;
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
    contextBlocks.push(r.document);
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

async function callLLM(messages, maxTokens = 1024) {
  const apiKey = process.env.OPENROUTER_API_KEY;
  if (!apiKey) {
    throw new Error('OPENROUTER_API_KEY is not configured.');
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
        'HTTP-Referer': 'https://recipe-chat.netlify.app',
        'X-Title': 'ResepAI',
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages,
        temperature: 0.7,
        max_tokens: maxTokens,
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
  const msg = message.toLowerCase().trim();

  // ===== DISCUSSION MODE: Simple chit-chat (no LLM needed) =====
  // Check if user is just greeting or making small talk
  const isGreeting = ['halo', 'hi', 'hey', 'p', 'woi', 'wooi', 'hola', 'hello'].some(g => msg === g || msg.startsWith(g + ' '));
  const isThanks = ['thanks', 'thank you', 'makasih', 'terima kasih', 'thx', 'ty'].some(t => msg.includes(t));
  const isReady = ['sudah', 'gas', 'skip', 'siap', 'langsung', 'cukup', 'ready', 'yup', 'langsung aja'].some(s => msg === s || msg === s + '!' || msg === s + '.');

  if (isGreeting) {
    const greetings = [
      'Halo! 😊 Aku ResepAI, temen ngobrol soal masak. Kamu mau bikin apa hari ini?',
      'Hai! 🍳 Ada yang bisa dibantu soal masak-masak?',
      'Halo halo! 😋 Kamu punya bahan apa aja di rumah?',
    ];
    return { reply: greetings[Math.floor(Math.random() * greetings.length)], sources: [], lowConfidence: true };
  }

  if (isThanks) {
    return { reply: 'Sama-sama! 😊 Ada lagi yang bisa dibantu?', sources: [], lowConfidence: true };
  }

  if (isReady) {
    // User is ready — switch to recipe mode (fall through to LLM below)
  } else if (!isRecipeRequest(msg)) {
    // Not a recipe request — use template discussion
    const hasIngredients = msg.includes('punya') || msg.includes('ada') || msg.includes('bahan');
    const isAskingIdea = msg.includes('ide') || msg.includes('bikin') || msg.includes('masak') || msg.includes('resep') || msg.includes('masakan');

    let reply;
    if (hasIngredients || isAskingIdea) {
      const questions = [
        'Wah menarik! 🍳 Kamu mau bikin yang gimana? Goreng, tumis, atau berkuah?',
        'Oke! 😋 Kamu mau yang simpel atau yang agak ribet? Dan buat berapa orang?',
        'Hmm, bisa banget! 🔥 Kamu mau yang pedas, manis, atau gurih?',
        'Siap! 🍳 Kamu punya bumbu apa aja di rumah? Biar aku sesuaikan resepnya.',
      ];
      reply = questions[Math.floor(Math.random() * questions.length)];
    } else {
      const general = [
        'Wah, aku penasaran! 🍳 Kamu mau bikin apa?',
        'Oke! 😊 Ceritain dong, kamu punya bahan apa aja?',
        'Hmm, menarik! 🔥 Kamu mau masak yang gimana?',
      ];
      reply = general[Math.floor(Math.random() * general.length)];
    }
    log('Discussion mode: template reply');
    return { reply, sources: [], lowConfidence: true };
  }

  // ===== RECIPE MODE: LLM with tool calling =====
  log('Recipe mode: LLM with tools');

  // Build initial messages
  let messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    ...safeHistory.map(m => ({ role: m.role, content: m.content })),
    { role: 'user', content: message.slice(0, MAX_MESSAGE_LENGTH) },
  ];

  // Tool-calling loop (max 2 iterations: search_recipe → google_search)
  let reply = '';
  for (let i = 0; i < 2; i++) {
    log(`LLM call ${i + 1}`);
    const llmOutput = await callLLM(messages, i === 0 ? 1024 : 2048);

    // Check if LLM wants to use a tool
    const toolMatch = llmOutput.match(/\{TOOL:\s*(search_recipe|google_search)\}\s*\nquery:\s*(.+?)\s*\{TOOL:\s*end\}/s);

    if (toolMatch) {
      const toolName = toolMatch[1];
      const query = toolMatch[2].trim();
      log(`Tool call: ${toolName}("${query}")`);

      let toolResult;
      if (toolName === 'search_recipe') {
        toolResult = await executeSearchRecipe(query);
      } else if (toolName === 'google_search') {
        toolResult = await executeGoogleSearch(query);
      }

      // Add tool result to messages and continue loop
      messages.push({ role: 'assistant', content: llmOutput });
      messages.push({ role: 'user', content: `Tool result:\n${toolResult}\n\nNow generate a conversational reply based on this information.` });
    } else {
      // No tool call — this is the final reply
      reply = llmOutput;
      break;
    }
  }

  // If loop ended without reply, get final response
  if (!reply) {
    reply = await callLLM(messages, 2048);
  }

  // Post-process
  reply = postProcessReply(reply);

  return { reply, sources: [], lowConfidence: false };
}

// Helper: Detect if message is a recipe request
function isRecipeRequest(msg) {
  const recipeKeywords = [
    'resep', 'masak', 'masakan', 'bikin', 'buat', 'cara', 'tutorial',
    'recipe', 'cook', 'how to', 'buat cara', 'cara membuat', 'cara bikin',
    'bahan', 'ingredient', 'bumbu', 'langkah', 'step',
  ];
  return recipeKeywords.some(k => msg.includes(k));
}

// Helper: Execute search_recipe tool (HF embeddings)
async function executeSearchRecipe(query) {
  try {
    const { context } = await retrieveContext(query);
    if (!context || context.trim().length < 10) {
      return 'No matching recipes found in database.';
    }
    return context;
  } catch (err) {
    log('search_recipe failed:', err.message);
    return 'Search failed. Try google_search instead.';
  }
}

// Helper: Execute google_search tool (DuckDuckGo)
async function executeGoogleSearch(query) {
  try {
    const searchUrl = `https://html.duckduckgo.com/html/?q=resep+${encodeURIComponent(query)}`;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);

    const response = await fetch(searchUrl, {
      headers: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
      },
      signal: controller.signal,
    });
    clearTimeout(timeout);

    if (!response.ok) {
      return 'Google search failed. Please try again.';
    }

    const html = await response.text();
    // Extract result snippets
    const results = [];
    const snippetRegex = /<a rel="nofollow" class="result__a"[^>]*>(.*?)<\/a>.*?<td class="result__snippet"[^>]*>(.*?)<\/td>/gs;
    let match;
    while ((match = snippetRegex.exec(html)) !== null && results.length < 5) {
      const title = match[1].replace(/<[^>]+>/g, '').trim();
      const snippet = match[2].replace(/<[^>]+>/g, '').trim();
      if (title && snippet) {
        results.push(`${title}: ${snippet}`);
      }
    }

    if (results.length === 0) {
      return 'No search results found.';
    }
    return results.join('\n\n');
  } catch (err) {
    log('google_search failed:', err.message);
    return 'Google search failed. Please try again.';
  }
}

// Helper: Post-process LLM reply
function postProcessReply(reply) {
  // Strip database-like phrases
  const dbPhrases = [
    'dari resep yang ada', 'dari database', 'dari sumber', 'menurut resep',
    'berdasarkan data', 'saya menemukan', 'saya mencari', 'berdasarkan resep',
    'dari informasi yang ada', 'dari data yang ada', 'menurut data',
    'dari konteks', 'konteks resep', 'konteks yang ada', 'berdasarkan konteks',
    'dari hasil', 'saya temukan', 'saya dapat', 'pencarian', 'mencari resep',
    'dari referensi', 'referensi yang saya', 'yang saya punya',
    'berikut salah satu', 'berikut ini', 'yang paling dekat',
    'yang cocok adalah', 'yang bisa kamu', 'yang bisa kalian',
    'resep yang paling dekat', 'yang paling cocok',
  ];

  const lowerReply = reply.toLowerCase();
  for (const phrase of dbPhrases) {
    const idx = lowerReply.indexOf(phrase);
    if (idx >= 0) {
      let sentenceStart = idx;
      while (sentenceStart > 0 && reply[sentenceStart - 1] !== '\n' && reply[sentenceStart - 1] !== '.') {
        sentenceStart--;
      }
      reply = reply.substring(0, sentenceStart).trim();
      break;
    }
  }

  // Strip "Tentu" / "Tentu saja" opening
  reply = reply.replace(/^(Tentu,?\s*(saja,?\s*)?)/i, '').trim();

  // If reply too short, add fallback
  if (reply.length < 20) {
    reply = 'Wah, menarik nih! 🍳 Ini resep yang aku temukan untuk kamu. Mau aku jelasin lebih detail? 😊';
  }

  // Inject conversational opening if needed
  const firstLine = reply.split('\n')[0].toLowerCase();
  const needsGreeting = !firstLine.includes('wah') && !firstLine.includes('oke') && !firstLine.includes('halo') && !firstLine.includes('hi') && !firstLine.includes('😊') && !firstLine.includes('🍳');
  if (needsGreeting && reply.length > 30) {
    const greetings = [
      'Wah, enak nih! 🍳\n\n',
      'Oke, ini resepnya ya! 😊\n\n',
      'Siap! Ini yang aku rekomendasiin 🔥\n\n',
    ];
    reply = greetings[Math.floor(Math.random() * greetings.length)] + reply;
  }

  // Add follow-up question if missing
  const lastLine = reply.split('\n').pop().toLowerCase();
  const hasQuestion = lastLine.includes('?') || lastLine.includes('gimana') || lastLine.includes('mau') || lastLine.includes('kamu');
  if (!hasQuestion && reply.length > 50) {
    const followUps = [
      '\n\nMau aku jelasin lebih detail? 😊',
      '\n\nGimana, cocok nggak? Atau mau yang lain? 😋',
      '\n\nAda bumbu tertentu yang kamu suka? 🔥',
    ];
    reply = reply + followUps[Math.floor(Math.random() * followUps.length)];
  }

  return reply;
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
