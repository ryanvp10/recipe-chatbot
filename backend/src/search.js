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
 * Call once at startup.
 */
function loadEmbeddings() {
  console.log('[search] Loading metadata...');
  const metaRaw = fs.readFileSync(METADATA_PATH, 'utf-8');
  const meta = JSON.parse(metaRaw);
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
}

/**
 * Search for top-K most similar embeddings to the query.
 * Uses direct buffer reads for performance (avoids creating intermediate arrays).
 * @param {number[]} queryEmbedding - 384-dimension normalized vector
 * @param {number} topK - number of results to return
 * @returns {Array<{id: string, document: string, metadata: object, similarity: number}>}
 */
function search(queryEmbedding, topK = 5) {
  if (!embeddingsBuffer) {
    throw new Error('Embeddings not loaded. Call loadEmbeddings() first.');
  }

  if (queryEmbedding.length !== embeddingDim) {
    throw new Error(
      `Query embedding dimension mismatch: expected ${embeddingDim}, got ${queryEmbedding.length}`
    );
  }

  // Compute similarities using direct buffer reads (avoid creating arrays)
  const similarities = new Float32Array(embeddingCount);
  for (let i = 0; i < embeddingCount; i++) {
    const offset = i * embeddingDim * 4;
    let dot = 0;
    for (let j = 0; j < embeddingDim; j++) {
      dot += queryEmbedding[j] * embeddingsBuffer.readFloatLE(offset + j * 4);
    }
    similarities[i] = dot;
  }

  // Find top-K using partial sort (min-heap approach for efficiency)
  const topResults = [];
  for (let i = 0; i < embeddingCount; i++) {
    const sim = similarities[i];
    if (topResults.length < topK) {
      topResults.push({ index: i, similarity: sim });
      if (topResults.length === topK) {
        topResults.sort((a, b) => a.similarity - b.similarity);
      }
    } else if (sim > topResults[0].similarity) {
      topResults[0] = { index: i, similarity: sim };
      topResults.sort((a, b) => a.similarity - b.similarity);
    }
  }

  // Sort descending
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
