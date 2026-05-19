const fs = require('fs');
const path = require('path');

const DATA_DIR = path.join(__dirname, '..', 'embeddings_data');
const METADATA_PATH = path.join(DATA_DIR, 'metadata.json');
const EMBEDDINGS_PATH = path.join(DATA_DIR, 'embeddings.bin');

let ids = [];
let documents = [];
let metadatas = [];
let embeddingCount = 0;
let embeddingDim = 0;
let embeddingsBuffer = null;

/**
 * Load embeddings from binary file + metadata JSON.
 * Call once at startup. Returns a Promise for async compatibility.
 */
function loadEmbeddings() {
  console.log('[search] Loading metadata...');

  if (!fs.existsSync(METADATA_PATH)) {
    throw new Error(`Metadata file not found: ${METADATA_PATH}`);
  }
  if (!fs.existsSync(EMBEDDINGS_PATH)) {
    throw new Error(`Embeddings file not found: ${EMBEDDINGS_PATH}`);
  }

  const metaRaw = fs.readFileSync(METADATA_PATH, 'utf-8');
  let meta;
  try {
    meta = JSON.parse(metaRaw);
  } catch (e) {
    throw new Error(`Failed to parse metadata JSON: ${e.message}`);
  }

  if (!meta.ids || !meta.documents || !meta.metadatas || !meta.count || !meta.dim) {
    throw new Error('Metadata JSON missing required fields (ids, documents, metadatas, count, dim)');
  }
  if (meta.ids.length !== meta.count || meta.documents.length !== meta.count || meta.metadatas.length !== meta.count) {
    throw new Error('Metadata array lengths do not match count field');
  }

  ids = meta.ids;
  documents = meta.documents;
  metadatas = meta.metadatas;
  embeddingCount = meta.count;
  embeddingDim = meta.dim;

  console.log(`[search] Loading embeddings binary (${embeddingCount}x${embeddingDim})...`);
  embeddingsBuffer = fs.readFileSync(EMBEDDINGS_PATH);

  const expectedBytes = embeddingCount * embeddingDim * 4; // float32 = 4 bytes
  if (embeddingsBuffer.length !== expectedBytes) {
    throw new Error(
      `Embeddings binary size mismatch: expected ${expectedBytes} bytes, got ${embeddingsBuffer.length}`
    );
  }

  console.log(`[search] Loaded ${embeddingCount} embeddings with ${embeddingDim} dimensions.`);
  return Promise.resolve();
}

/**
 * Search for top-K most similar embeddings to the query.
 * Uses direct buffer reads for performance.
 * @param {number[]} queryEmbedding - 384-dimension normalized vector
 * @param {number} topK - number of results to return (1-100)
 * @returns {Array<{id: string, document: string, metadata: object, similarity: number}>}
 */
function search(queryEmbedding, topK = 5) {
  if (!embeddingsBuffer) {
    throw new Error('Embeddings not loaded. Call loadEmbeddings() first.');
  }

  if (!Array.isArray(queryEmbedding) || queryEmbedding.length !== embeddingDim) {
    throw new Error(
      `Query embedding must be an array of length ${embeddingDim}, got ${Array.isArray(queryEmbedding) ? queryEmbedding.length : typeof queryEmbedding}`
    );
  }

  topK = Math.max(1, Math.min(topK, 100));

  const topResults = [];
  for (let i = 0; i < embeddingCount; i++) {
    const offset = i * embeddingDim * 4;
    let dot = 0;
    for (let j = 0; j < embeddingDim; j++) {
      dot += queryEmbedding[j] * embeddingsBuffer.readFloatLE(offset + j * 4);
    }

    if (topResults.length < topK) {
      topResults.push({ index: i, similarity: dot });
      if (topResults.length === topK) {
        topResults.sort((a, b) => a.similarity - b.similarity);
      }
    } else if (dot > topResults[0].similarity) {
      topResults[0] = { index: i, similarity: dot };
      topResults.sort((a, b) => a.similarity - b.similarity);
    }
  }

  topResults.sort((a, b) => b.similarity - a.similarity);

  return topResults.map((r) => ({
    id: ids[r.index],
    document: documents[r.index],
    metadata: metadatas[r.index],
    similarity: r.similarity,
  }));
}

/**
 * Get total number of loaded embeddings.
 * @returns {number}
 */
function getEmbeddingCount() {
  return embeddingCount;
}

module.exports = {
  loadEmbeddings,
  search,
  getEmbeddingCount,
};
