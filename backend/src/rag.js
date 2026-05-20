const dotenv = require('dotenv');
const fetch = require('node-fetch');
const { loadEmbeddings, search, getEmbeddingCount } = require('./search');

dotenv.config();

const LLM_URL = 'https://openrouter.ai/api/v1/chat/completions';
const HF_EMBEDDING_URL = 'https://router.huggingface.co/hf-inference/v1/pipeline/feature-extraction/BAAI/bge-small-en-v1.5';
const MODEL_NAME = 'openrouter/owl-alpha';
const SYSTEM_PROMPT =
  'Kamu adalah ResepAI. Tugasmu: bantu user soal masak dan resep.\n\nATURAN:\n- Jawab SANTAI, kayak chat sama temen. Pakai "kamu".\n- Selalu pakai emoji: 😊🍳👍🔥😋✨\n- JANGAN formal, jangan kaku, jangan bilang "menurut", "berdasarkan", "dari data", "dari resep yang ada", "konteks", "pencarian".\n- Langsung kasih resep natural, jangan format kaku.\n- Kalau di luar topik masak: "Maaf, aku cuma bisa bantu soal masak-masak 😊"';
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
    // Discussion mode: ALWAYS ask questions, NEVER give recipes
    const chatPrompt = `Kamu adalah temen ngobrol soal masak. Santai, natural, kayak chat sama temen. Pakai "kamu", emoji secukupnya 😊🍳.

ATURAN PENTING — WAJIB DIIKUTI:
- KAMU TIDAK BOLEH KASIH RESEP. PUN. Dalam kondisi apapun. Jangan kasih bahan, jangan kasih langkah, jangan kasih resep lengkap. PUNYA BAHAN APA AJA.
- Tugasmu CUMA tanya balik ke user. Tanya terus sampai user bilang "sudah/gas/skip/siap/langsung aja/cukup/ready".
- Kalau user kasih daftar bahan, JANGAN langsung kasih resep. Tanya dulu: "Mau bikin apa?", "Yang gimana?", "Ada preferensi tertentu?"
- Minimal tanya 2-3 hal sebelum user boleh dapet resep.
- Jawab pendek, 2-3 kalimat max.
- Kalau user bilang "sudah/gas/skip/siap/langsung aja/cukup", bilang: "Oke siap! Sebentar ya, aku siapin resepnya 🍳" — TETAP jangan kasih resep, biarkan sistem yang kasih.

Contoh flow yang BENAR:
User: "Aku punya ayam, lada, bawang"
Bot: "Wah simpel tapi enak nih! 🍗 Kamu mau bikin ayam goreng, bakar, atau yang lain?"

User: "Ayam goreng"
Bot: "Oke ayam goreng! 🔥 Kamu mau yang krispy atau yang biasa aja? Dan buat berapa orang?"

User: "Krispy, buat 2 orang"
Bot: "Siap! Aku siapin resepnya ya 🍳"

Contoh flow yang SALAH (JANGAN LAKUKAN):
User: "Aku punya ayam, lada, bawang"
Bot: "Ini resep ayam goreng: ..." ❌`;

    const messages = [
      { role: 'system', content: chatPrompt },
      ...safeHistory.map(m => ({ role: m.role, content: m.content })),
      { role: 'user', content: message.slice(0, MAX_MESSAGE_LENGTH) },
    ];

    log('Discussion mode: chatting');
    let reply = await callLLM(messages, 200);

    // Post-process: strip any recipe content that LLM might have added
    const lowerReply = reply.toLowerCase();
    const cutPoints = [
      'berikut resep', 'berikut ini', 'berikut adalah', 'ini resep', 'ini dia resep',
      '## ', '### ', '**bahan', '**cara', '**langkah', '**steps', '**ingredients', '**tips',
      'bahan:', 'cara membuat:', 'langkah:', '1. ', '2. ', '3. ',
      'bahan-bahan:', 'cara pembuatan:', 'langkah-langkah:',
      'siapkan bahan', 'pertama-tama', 'langkah pertama',
    ];
    for (const marker of cutPoints) {
      const idx = lowerReply.indexOf(marker);
      if (idx >= 0) {
        reply = reply.substring(0, idx).trim();
        break;
      }
    }

    // If reply is too short after cleanup, provide a fallback question
    if (reply.length < 15) {
      const fallbacks = [
        'Wah menarik! 🍳 Kamu mau bikin yang gimana?',
        'Oke! Coba ceritain lagi, kamu mau masak apa? 😊',
        'Hmm, aku penasaran — kamu mau bikin apa nih? 🍳',
      ];
      reply = fallbacks[Math.floor(Math.random() * fallbacks.length)];
    }

    // Fallback if reply is too short after cleanup
    if (reply.length < 15) {
      reply = `Oke! ${message} ya? Mau yang gimana? Yang simpel atau yang lengkap? 😊`;
    }

    return { reply, sources: [], lowConfidence: true };
  }

  // Recipe mode: retrieve context and give full recipe
  const { context, sources, lowConfidence } = await retrieveContext(message);

  const systemContent = [
    SYSTEM_PROMPT,
    'MODE: RESEP LENGKAP. User sudah minta resep. Langsung kasih resep dengan gaya santai, pakai emoji, kayak temen yang lagi share resep. JANGAN formal. JANGAN bilang "menurut", "berdasarkan", "dari data".',
    `Contoh jawaban yang BENAR:\n"Wah, tahu-tempe enak nih! 🍳 Ini resep tahu kecap yang simpel dan enak:\n\nKamu butuh: tahu putih, tempe, bawang merah, bawang putih, kecap manis, cabe, garam, gula.\n\nCaranya: goreng tahu dan tempe sampai kecoklatan. Tumis bawang dan cabe, tambah air, kecap manis, garam, gula. Masukkan tahu dan tempe, masak sampai bumbu meresap. Sajikan! 😋"`,
    `Contoh jawaban yang SALAH (JANGAN):\n"Berikut ide masakan dari konteks resep yang ada: Opor ayam kuning tanpa MSG. Bahan: ..."`,
    `Info resep untuk referensi:\n${context || 'Tidak ada info tambahan.'}`,
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
    'dari konteks', 'konteks resep', 'berdasarkan konteks', 'dari hasil',
    'saya temukan', 'saya dapat', 'pencarian', 'mencari resep',
    'dari referensi', 'referensi yang saya', 'yang saya punya',
    'berikut salah satu', 'berikut ini', 'yang paling dekat',
    'yang cocok adalah', 'yang bisa kamu', 'yang bisa kalian',
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

  // If reply got too short after stripping, regenerate with a simpler fallback
  if (reply.length < 20) {
    reply = 'Oke, ini resepnya ya! 🍳\n\n' + (context ? context.split('\n').slice(0, 30).join('\n') : 'Maaf, aku butuh info lebih lanjut.');
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
