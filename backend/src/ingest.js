const fetch = require('node-fetch');
const { ChromaClient } = require('chromadb');
const { pipeline } = require('@xenova/transformers');

const DATASET_NAME = 'junwatu/indonesian-recipes';
const DATASET_CONFIG = 'default';
const DATASET_SPLIT = 'train';
const DEFAULT_TEST_BATCH = 10;
const DEFAULT_BATCH_SIZE = 100;
const COLLECTION_NAME = 'recipes';
const CHROMA_HOST = process.env.CHROMA_HOST || 'localhost';
const CHROMA_PORT = process.env.CHROMA_PORT || 8000;
const CHROMA_URL = `http://${CHROMA_HOST}:${CHROMA_PORT}`;

let extractorPromise;
let chromaClientPromise;

function log(...args) {
  console.log('[ingest]', ...args);
}

function warn(...args) {
  console.warn('[ingest]', ...args);
}

async function getEmbeddingPipeline() {
  if (!extractorPromise) {
    extractorPromise = pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2');
  }
  return extractorPromise;
}

async function embedTexts(texts) {
  const extractor = await getEmbeddingPipeline();
  const vectors = [];

  for (const text of texts) {
    const output = await extractor(text, { pooling: 'mean', normalize: true });
    vectors.push(Array.from(output.data));
  }

  return vectors;
}

function getChromaClient() {
  if (!chromaClientPromise) {
    chromaClientPromise = Promise.resolve(
      new ChromaClient({ host: CHROMA_HOST, port: CHROMA_PORT })
    );
  }
  return chromaClientPromise;
}

async function getOrCreateCollection() {
  const client = await getChromaClient();
  return client.getOrCreateCollection({ name: COLLECTION_NAME });
}

async function getCollectionCount(collection) {
  try {
    return await collection.count();
  } catch (error) {
    warn('Unable to count collection documents.', error.message);
    return 0;
  }
}

function normalizeArrayField(value) {
  if (Array.isArray(value)) {
    return value.filter(Boolean).map((item) => String(item).trim()).filter(Boolean);
  }

  if (typeof value === 'string') {
    return value
      .split(/\r?\n|;|\|/)
      .map((item) => item.trim())
      .filter(Boolean);
  }

  return [];
}

function getRowPayload(row) {
  if (row && typeof row === 'object' && row.row && typeof row.row === 'object') {
    return row.row;
  }
  return row;
}

function buildRecipeDocument(row, index) {
  const payload = getRowPayload(row) || {};
  const title = String(payload.title || payload.name || payload.recipe_name || `Recipe ${index + 1}`).trim();
  const ingredients = normalizeArrayField(payload.ingredients || payload.ingredient || payload.bahan);
  const steps = normalizeArrayField(payload.steps || payload.step || payload.instructions || payload.langkah);

  const pageContent = [
    `Title: ${title}`,
    `Ingredients: ${ingredients.join(', ') || 'N/A'}`,
    `Steps: ${steps.join(' | ') || 'N/A'}`,
  ].join('\n');

  return {
    id: `recipe-${payload.id || payload._id || payload.slug || index}`,
    document: pageContent,
    metadata: {
      title,
      num_ingredients: ingredients.length,
      num_steps: steps.length,
      source: 'indonesian-recipes',
    },
  };
}

async function fetchDatasetRows(offset, length) {
  const url = `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET_NAME)}&config=${encodeURIComponent(DATASET_CONFIG)}&split=${encodeURIComponent(DATASET_SPLIT)}&offset=${offset}&length=${length}`;
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Dataset fetch failed with status ${response.status}`);
  }

  const data = await response.json();
  return {
    rows: Array.isArray(data.rows) ? data.rows : [],
    total: typeof data.num_rows_total === 'number' ? data.num_rows_total : null,
  };
}

async function addRecipesToCollection(collection, recipes) {
  const ids = recipes.map((recipe) => recipe.id);
  const documents = recipes.map((recipe) => recipe.document);
  const metadatas = recipes.map((recipe) => recipe.metadata);
  const embeddings = await embedTexts(documents);

  await collection.add({ ids, documents, metadatas, embeddings });
}

async function ingestRecipes(options = {}) {
  const testBatchSize = Number(options.testBatchSize || process.env.INGEST_TEST_BATCH || DEFAULT_TEST_BATCH);
  const batchSize = Number(options.batchSize || process.env.INGEST_BATCH_SIZE || DEFAULT_BATCH_SIZE);
  const collection = await getOrCreateCollection();
  const existingCount = await getCollectionCount(collection);

  if (existingCount > 0) {
    log(`Collection already contains ${existingCount} documents. Skipping ingestion.`);
    return { skipped: true, count: existingCount };
  }

  log(`Starting dataset ingestion for ${DATASET_NAME}.`);
  log(`Running small-batch ingestion test with ${testBatchSize} recipes.`);

  const testResult = await fetchDatasetRows(0, testBatchSize);
  const testRecipes = testResult.rows.map(buildRecipeDocument);
  if (testRecipes.length === 0) {
    throw new Error('Dataset returned no rows during ingest test.');
  }

  await addRecipesToCollection(collection, testRecipes);
  let ingestedCount = testRecipes.length;
  const totalRows = testResult.total || testRecipes.length;
  log(`Ingested ${ingestedCount}/${totalRows} recipes (test batch).`);

  for (let offset = ingestedCount; offset < totalRows; offset += batchSize) {
    const currentBatchSize = Math.min(batchSize, totalRows - offset);
    const result = await fetchDatasetRows(offset, currentBatchSize);
    const recipes = result.rows.map((row, index) => buildRecipeDocument(row, offset + index));

    if (recipes.length === 0) {
      warn(`No rows returned for offset ${offset}. Stopping ingestion.`);
      break;
    }

    await addRecipesToCollection(collection, recipes);
    ingestedCount += recipes.length;
    log(`Ingested ${ingestedCount}/${totalRows} recipes.`);
  }

  log(`Ingestion complete. Stored ${ingestedCount} recipes in collection '${COLLECTION_NAME}'.`);
  return { skipped: false, count: ingestedCount };
}

module.exports = {
  COLLECTION_NAME,
  CHROMA_URL,
  embedTexts,
  getChromaClient,
  getCollectionCount,
  getOrCreateCollection,
  ingestRecipes,
};

if (require.main === module) {
  ingestRecipes().catch((error) => {
    console.error('[ingest] Fatal error during ingestion:', error);
    process.exit(1);
  });
}
