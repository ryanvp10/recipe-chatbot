const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const LLM_URL = 'https://api.freemodel.dev/v1/chat/completions';
const HF_EMBEDDING_URL = 'https://router.huggingface.co/hf-inference/v1/pipeline/feature-extraction/BAAI/bge-small-en-v1.5';
const MODEL_NAME = 'gpt-5.5';
const SYSTEM_PROMPT = `Kamu adalah ResepAI, teman ngobrol soal masak-masak. 

ATURAN:
- Bahasa Indonesia santai, pakai "kamu"
- Pakai emoji: 😊🍳🔥😋✨👍
- Langsung jawab, jangan pakai pembuka seperti "Tentu", "Berikut adalah", "Berdasarkan"
- JANGAN PERNAH sebut: database, konteks, sumber, referensi, data, pencarian, ditemukan
- JANGAN PERNAH pakai format "ada X jenis yang terdetekti"
- Langsung kasih resep dengan format:

🍳 [Nama]

📋 Bahan:
• ...

👨‍🍳 Cara:
1. ...

💡 Tips: ...

- Selalu akhiri dengan pertanyaan balik pakai emoji
- Kalau nggak bisa bantu: "Maaf, aku cuma bisa bantu soal masak-masak 😊"
- KHUSUS RESEP: Kasih SATU resep terbaik saja, jangan multiple. Format: 🍳 Nama, 📋 Bahan, 👨‍🍳 Cara, 💡 Tips, lalu SATU pertanyaan balik.

Kamu punya alat:
{TOOL: search_recipe}
query: [pencarian]
{TOOL: end}

Setelah dapat hasil alat, langsung kasih resep dengan format di atas.`;
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
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model: MODEL_NAME,
        messages: messages,
        max_tokens: maxTokens,
        temperature: 0.7,
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

  // ===== CHECK: Ask clarifying question if no preference specified =====
  const userHasPreference = hasUserPreference(message, safeHistory);
  if (!userHasPreference) {
    const ingredient = message.match(/(ayam|ikan|tahu|tempe|telur|daging|udang|sayur)/i);
    const bahan = ingredient ? ingredient[1] : 'bahan';
    const questions = [
      `Wah, ${bahan} enak tuh! 🍳 Kamu mau digoreng, ditumis, atau dibikin kuah?`,
      `Oke! 😋 Kamu suka yang pedas, manis, atau gurih?`,
      `Bisa banget! 🔥 Kamu mau yang simpel cepat atau yang agak ribet?`,
    ];
    const reply = questions[Math.floor(Math.random() * questions.length)];
    log('Discussion mode: asking clarifying question');
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
  reply = postProcessReply(reply, message);

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

// Helper: Check if user has specified a cooking preference
function hasUserPreference(message, history) {
  const preferenceKeywords = [
    'goreng', 'tumis', 'bakar', 'kukus', 'rebus', 'kuah', 'sup', 'soto',
    'pedas', 'manis', 'gurih', 'asam', 'segar', 'asin',
    'simpel', 'cepat', 'ribet', 'mudah', 'gampang',
    'berkuah', 'kering', 'soup', 'stir-fry', 'fried'
  ];
  const msg = message.toLowerCase();
  // Check current message
  if (preferenceKeywords.some(k => msg.includes(k))) return true;
  // Check last 3 messages in history
  const recentHistory = history.slice(-3);
  for (const h of recentHistory) {
    const hMsg = h.content.toLowerCase();
    if (preferenceKeywords.some(k => hMsg.includes(k))) return true;
  }
  return false;
}

// Helper: Execute search_recipe tool (HF embeddings)
async function executeSearchRecipe(query) {
  try {
    const { context } = await retrieveContext(query);
    if (!context || context.trim().length < 10) {
      return 'Nggak nemu resep yang cocok nih. Coba kata kunci lain ya!';
    }
    return context;
  } catch (err) {
    log('search_recipe failed:', err.message);
    return 'Pencarian gagal. Coba lagi ya!';
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
      return 'Pencarian gagal. Coba lagi ya!';
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
      return 'Nggak nemu hasil pencarian.';
    }
    return results.join('\n\n');
  } catch (err) {
    log('google_search failed:', err.message);
    return 'Pencarian gagal. Coba lagi ya!';
  }
}

// Helper: Post-process LLM reply
function postProcessReply(reply, userQuery = '') {
  // === STEP 1: Strip entire first paragraph if it contains robotic openers ===
  const roboticOpeners = [
    /dari konteks/i, /dari resep/i, /tentu/i, /berikut adalah/i,
    /berikut ini/i, /ada\s+\d+\s+jenis/i, /yang terdeteksi/i,
  ];
  const paragraphs = reply.split(/\n\n+/);
  if (paragraphs.length > 1) {
    const firstPara = paragraphs[0].toLowerCase();
    const hasRoboticOpener = roboticOpeners.some(r => r.test(firstPara));
    if (hasRoboticOpener) {
      paragraphs.shift(); // Remove entire first paragraph
      reply = paragraphs.join('\n\n').trim();
    }
  }

  // === STEP 2: Remove lines containing forbidden phrases ===
  const forbiddenLines = [
    /terdeteksi/i, /ditemukan/i, /pencarian/i, /database/i,
    /konteks/i, /sumber/i, /referensi/i, /menurut/i,
    /berdasarkan data/i, /ada\s+\d+\s+jenis/i, /yang cocok adalah/i,
    /yang paling cocok/i, /resep yang paling/i,
    /dari data/i, /dari hasil/i, /saya temukan/i, /saya dapat/i,
    /saya menemukan/i, /saya mencari/i, /saya punya/i,
    /berikut salah satu/i, /yang bisa kamu/i, /yang bisa kalian/i,
    /database tidak ada/i, /persis sama/i, /saran masak umum/i,
    /ini saran/i, /dari informasi yang ada/i, /dari referensi/i,
    /referensi yang saya/i, /yang saya punya/i,
  ];
  const lines = reply.split('\n');
  const filteredLines = lines.filter(line => {
    const lowerLine = line.toLowerCase().trim();
    if (lowerLine.length === 0) return true; // keep blank lines
    return !forbiddenLines.some(r => r.test(lowerLine));
  });
  reply = filteredLines.join('\n').trim();

  // === STEP 3: If reply too short after stripping, replace with template ===
  if (reply.length < 100) {
    const query = userQuery || 'masakan';
    const templates = [
      `Wah, ${query} ya? Aku bantu cariin resepnya ya! 🍳\n\nTapi kayaknya databasanya belum lengkap nih. Coba tanya yang lebih spesifik, misalnya "resep ${query} gampang" atau "cara bikin ${query}" 🔥\n\nAtau kamu mau aku carikan resep lain dulu? 😊`,
      `Hmm, ${query}! Enak tuh 😋\n\nSayangnya aku belum nemu resep yang pas di database. Coba deh tanya dengan kata kunci lain, atau bilang aja "cara membuat ${query}" 🍳\n\nMau coba yang lain? ✨`,
      `Oke, ${query}! 🔥\n\nAku lagi cariin resepnya tapi belum ketemu yang pas. Coba kamu spesifikasi lagi, misalnya bahan yang kamu punya atau cara masaknya gimana?\n\nAku siap bantu! 😊`,
    ];
    reply = templates[Math.floor(Math.random() * templates.length)];
    return reply;
  }

  // === STEP 4: Strip numbered list introductions → convert to 🍳 header ===
  // e.g. "1. Sate Ayam Manis" → "🍳 Sate Ayam Manis"
  reply = reply.replace(/^\d+\.\s+(.+)$/gm, (match, name) => {
    // Only convert if it looks like a recipe name (short line, no period at end)
    if (name.length < 60 && !name.endsWith('.')) {
      return `🍳 ${name}`;
    }
    return match;
  });

  // === STEP 5: Ensure emoji in section headers ===
  reply = reply.replace(/^Bahan:/gm, '📋 Bahan:');
  reply = reply.replace(/^Langkah:/gm, '👨‍🍳 Langkah:');
  reply = reply.replace(/^Cara membuat:/gm, '👨‍🍳 Cara membuat:');
  reply = reply.replace(/^Cara:/gm, '👨‍🍳 Cara:');
  reply = reply.replace(/^Tips:/gm, '💡 Tips:');
  reply = reply.replace(/^Nama:/gm, '🍳');

  // === STEP 6: Ensure last line is a question with emoji ===
  const lastLine = reply.split('\n').pop().trim();
  const hasQuestion = lastLine.includes('?') || /gimana|mau|bisa|ada|mau\s|coba|atau|yuk/i.test(lastLine);
  const hasEmoji = /[\u{1F300}-\u{1F9FF}]/u.test(lastLine);
  if (!hasQuestion || !hasEmoji) {
    const followUps = [
      '\n\nMau aku jelasin lebih detail? 😊',
      '\n\nGimana, cocok nggak? Atau mau yang lain? 😋',
      '\n\nAda bumbu tertentu yang kamu suka? 🔥',
      '\n\nMau coba resep ini? Atau ada yang mau ditanya? ✨',
      '\n\nKamu mau yang versi pedas atau yang biasa aja? 😊',
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
