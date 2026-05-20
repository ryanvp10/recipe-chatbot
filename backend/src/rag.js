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

  if (!confirmed) {
    // Discussion mode: Use template-based responses, don't rely on LLM
    const msg = message.toLowerCase().trim();

    // Check if user is ready for recipe
    const readySignals = ['sudah', 'gas', 'skip', 'siap', 'langsung', 'cukup', 'ready', 'yup', 'langsung aja'];
    if (readySignals.some(s => msg === s || msg === s + '!' || msg === s + '.')) {
      return { reply: 'Oke siap! Aku siapin resepnya ya 🍳', sources: [], lowConfidence: true };
    }

    // Check if user is listing ingredients
    const hasIngredients = msg.includes('punya') || msg.includes('ada') || msg.includes('bahan') || msg.includes('punya');
    const isAskingIdea = msg.includes('ide') || msg.includes('bikin') || msg.includes('masak') || msg.includes('resep');

    // Template responses based on context
    let reply;
    if (hasIngredients || isAskingIdea) {
      // User has ingredients or asking for ideas — ask follow-up
      const questions = [
        'Wah menarik! 🍳 Kamu mau bikin yang gimana? Goreng, tumis, atau berkuah?',
        'Oke! 😋 Kamu mau yang simpel atau yang agak ribet? Dan buat berapa orang?',
        'Hmm, bisa banget! 🔥 Kamu mau yang pedas, manis, atau gurih?',
        'Siap! 🍳 Kamu punya bumbu apa aja di rumah? Biar aku sesuaikan resepnya.',
        'Wah enak nih! 😊 Kamu mau yang cepat atau yang slow-cook?',
      ];
      reply = questions[Math.floor(Math.random() * questions.length)];
    } else {
      // General discussion — ask what they want to cook
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

  // Recipe mode: retrieve context and give full recipe
  const { context, sources, lowConfidence } = await retrieveContext(message);

  const systemContent = [
    SYSTEM_PROMPT,
    `<recipe_mode>
The user wants a full recipe. Use the reference info below to give ONE best recipe.
- Pick the SINGLE best match. Do NOT list multiple recipes.
- Be conversational: "Wah, [resep] enak nih! 🍳"
- Use the formatting rules from your system prompt.
- End with a follow-up question.
- NEVER say "dari referensi", "dari data", "menurut", "berdasarkan".
</recipe_mode>

<reference_info>
${context || 'No additional info available.'}
</reference_info>

<example_good>
Wah, tahu kecap simpel enak nih! 🍳

Kamu butuh: tahu putih, tempe, bawang merah, bawang putih, kecap manis, cabe, garam, gula.

Caranya: goreng tahu dan tempe sampai kecoklatan. Tumis bawang dan cabe, tambah air, kecap manis, garam, gula. Masukkan tahu dan tempe, masak sampai bumbu meresap. Sajikan! 😋

Mau yang pedas atau yang manis? 🔥
</example_good>

<example_bad>
Berikut ide masakan dari konteks resep yang ada: Opor Ayam Kuning. Bahan: ...
DO NOT REPLY LIKE THIS.
</example_bad>`,
  ].join('\n\n');

  const messages = [
    { role: 'system', content: systemContent },
    ...safeHistory,
    { role: 'user', content: message.slice(0, MAX_MESSAGE_LENGTH) },
  ];

  log('Generating reply with', sources.length, 'sources. Low confidence:', lowConfidence);
  let reply = await callLLM(messages);

  // Post-process: strip database-like phrases
  const dbPhrases = [
    'dari resep yang ada', 'dari database', 'dari sumber', 'menurut resep',
    'berdasarkan data', 'saya menemukan', 'saya mencari', 'berdasarkan resep',
    'dari informasi yang ada', 'dari data yang ada', 'menurut data',
    'dari konteks', 'konteks resep', 'konteks yang ada', 'berdasarkan konteks', 'dari hasil',
    'saya temukan', 'saya dapat', 'pencarian', 'mencari resep',
    'dari referensi', 'referensi yang saya', 'yang saya punya',
    'berikut salah satu', 'berikut ini', 'yang paling dekat',
    'yang cocok adalah', 'yang bisa kamu', 'yang bisa kalian',
    'resep yang paling dekat', 'yang paling cocok',
  ];
  const lowerReply = reply.toLowerCase();
  for (const phrase of dbPhrases) {
    const idx = lowerReply.indexOf(phrase);
    if (idx >= 0) {
      // Find the start of this sentence and remove from there
      let sentenceStart = idx;
      while (sentenceStart > 0 && reply[sentenceStart - 1] !== '\n' && reply[sentenceStart - 1] !== '.') {
        sentenceStart--;
      }
      reply = reply.substring(0, sentenceStart).trim();
      break;
    }
  }

  // If reply got too short after stripping, use a conversational fallback
  if (reply.length < 20) {
    // Try to extract recipe title from context
    const titleMatch = context?.match(/^([A-Z][^\n]+)/m);
    const recipeTitle = titleMatch ? titleMatch[1].trim() : 'resep ini';
    reply = `Wah, ${recipeTitle} enak nih! 🍳\n\n`;
    if (context) {
      // Take first 20 lines of context as recipe body
      const body = context.split('\n').slice(0, 20).join('\n');
      reply += body;
    }
    reply += '\n\nMau aku jelasin lebih detail? 😊';
  }

  // Inject conversational opening if reply starts with recipe title (no greeting)
  const firstLine = reply.split('\n')[0].toLowerCase();
  const needsGreeting = !firstLine.includes('wah') && !firstLine.includes('oke') && !firstLine.includes('halo') && !firstLine.includes('hi') && !firstLine.includes('😊') && !firstLine.includes('🍳');
  if (needsGreeting && reply.length > 30) {
    const greetings = [
      'Wah, enak nih! 🍳\n\n',
      'Oke, ini resepnya ya! 😊\n\n',
      'Siap! Ini yang aku rekomendasiin 🔥\n\n',
      'Mantap, ini resep yang pas buat kamu! 😋\n\n',
    ];
    reply = greetings[Math.floor(Math.random() * greetings.length)] + reply;
  }

  // Add follow-up question at the end if not already present
  const lastLine = reply.split('\n').pop().toLowerCase();
  const hasQuestion = lastLine.includes('?') || lastLine.includes('gimana') || lastLine.includes('mau') || lastLine.includes('kamu');
  if (!hasQuestion && reply.length > 50) {
    const followUps = [
      '\n\nKamu mau yang gimana? Yang simpel atau yang lengkap? 😊',
      '\n\nAda preferensi tertentu? Mau yang kuat atau yang ringan? 🍳',
      '\n\nGimana, cocok nggak? Atau mau yang lain? 😋',
      '\n\nMau aku jelasin lebih detail soal bumbunya? 🔥',
    ];
    reply = reply + followUps[Math.floor(Math.random() * followUps.length)];
  }

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
