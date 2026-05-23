const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const LLM_URL = 'https://api.freemodel.dev/v1/chat/completions';
const HF_EMBEDDING_URL = 'https://router.huggingface.co/hf-inference/v1/pipeline/feature-extraction/BAAI/bge-small-en-v1.5';
const MODEL_NAME = 'gpt-5.5';
const SYSTEM_PROMPT = `Kamu adalah ResepAI, temen ngobrol soal masak-masak.

ATURAN:
- Bahasa Indonesia santai, pakai "kamu" dan "aku"
- Pakai emoji: 😊🍳🔥😋✨👍
- JANGAN PERNAH sebut: database, konteks, sumber, referensi, data, pencarian, ditemukan
- JANGAN pakai pembuka seperti "Tentu", "Berikut adalah", "Berdasarkan"
- JANGAN pakai format "ada X jenis yang terdeteksi"
- Saat kasih daftar/pilihan apapun, VARIASI formatnya — kadang angka, kadang emoji berbeda per item, kadang bullet, kadang langsung paragraf. JANGAN selalu pakai format yang sama. Buat terasa natural seperti chat sama temen.

ALUR PERCAKAPAN:
- Baca seluruh riwayat percakapan sebelum menjawab
- Jika user minta PILIHAN/OPTIONS/IDE/REKOMENDASI (contoh: "kasih pilihan", "ada ide apa aja", "rekomendasiin"), kasih 3-5 NAMA RESEP saja dalam format daftar, jangan kasih resep lengkap. VARIASI format daftar — kadang pakai emoji berbeda per item, kadang angka, kadang bullet, jangan selalu sama. Contoh variasi:
  "Ini beberapa sambal yang enak:
  1. Sambal Bawang 🔥
  2. Sambal Terasi
  3. Sambal Tomat 🍅
  4. Sambal Ijo
  5. Sambal Korek 🌶️
  Mana yang kamu mau?"
- Jika user minta resep yang punya BANYAK VARIASI (contoh: "resep ayam goreng", "resep nasi goreng", "resep sambal", "resep rendang") DAN belum menyebut variasi spesifik, jangan langsung kasih resep lengkap. TANYA DULU variasi apa yang diinginkan, kasih 3-5 pilihan nama resep. Contoh:
  "Kamu lagi cari ayam goreng yang mana? 🍳
  1. Ayam Goreng Bawang
  2. Ayam Goreng Kuning
  3. Ayam Goreng Crispy
  4. Ayam Goreng Pedas
  Mana yang kamu mau?"
  Baru kasih resep LENGKAP setelah user memilih satu.
- Jika user sudah kasih cukup info (bahan + preferensi) atau memilih salah satu resep, kasih 1 resep LENGKAP
- Jika user menjawab pertanyaan kamu (misal: "pedas", "goreng", "simpel"), gunakan jawaban itu untuk kasih 1 resep LENGKAP
- Kasih resep lengkap hanya 1, jangan multiple
- Selalu akhiri dengan pertanyaan balik pakai emoji

FORMAT RESEP LENGKAP:
🍳 [Nama]

📋 Bahan:
• ...

👨‍🍳 Cara:
1. ...

💡 Tips: ...

ALAT:
{TOOL: search_recipe}
query: [pencarian]
{TOOL: end}

Setelah dapat hasil alat, ikuti ALUR PERCAKAPAN di atas: jika user minta pilihan, kasih daftar nama saja. Jika user sudah spesifik, kasih 1 resep lengkap.`;
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

async function generateRecipeReply(message, history = []) {
  const safeHistory = sanitizeHistory(history);

  // ===== ALL HANDLED BY LLM — pass full conversation history =====
  log('LLM mode: full conversation context');

  // Build messages: system prompt + full history + current message
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
      messages.push({ role: 'user', content: `Tool result:\n${toolResult}\n\nSekarang kasih resep dengan format yang sesuai.` });
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

  // === STEP 3: (removed — LLM handles all replies) ===

  // === STEP 4: (removed — LLM handles list formatting naturally) ===

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
