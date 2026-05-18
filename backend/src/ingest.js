require('dotenv').config();
const fetch = require('node-fetch');
const { ChromaClient } = require('chromadb');
const { pipeline } = require('@xenova/transformers');
const fs = require('fs');
const path = require('path');

const DATASET_NAME = 'junwatu/indonesian-recipes';
const DATASET_CONFIG = 'default';
const DATASET_SPLIT = 'train';
const DEFAULT_TEST_BATCH = 10;
const DEFAULT_BATCH_SIZE = 100;
const COLLECTION_NAME = 'recipes';
const CHROMA_HOST = process.env.CHROMA_HOST || 'localhost';
const CHROMA_PORT = process.env.CHROMA_PORT || 8000;
const CHROMA_URL = `http://${CHROMA_HOST}:${CHROMA_PORT}`;
const DATA_DIR = path.join(__dirname, '..', 'data');
const JSON_PATH = path.join(DATA_DIR, 'train.json');

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

function buildRecipeDocument(row, index) {
  const payload = row || {};
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

async function downloadDataset() {
  // Use huggingface_hub CLI to download the dataset
  const { execSync } = require('child_process');

  if (fs.existsSync(JSON_PATH)) {
    log('Dataset JSON already exists, skipping download.');
    return;
  }

  log('Downloading dataset from HuggingFace...');

  const hfToken = process.env.HF_TOKEN;
  if (!hfToken) {
    throw new Error('HF_TOKEN not set in .env');
  }

  // Use huggingface_hub Python library to download
  const pythonScript = `
from huggingface_hub import hf_hub_download
import os

token = os.environ.get('HF_TOKEN', '')
path = hf_hub_download(
    repo_id='${DATASET_NAME}',
    filename='data/train.parquet',
    repo_type='dataset',
    token=token,
    local_dir='${DATA_DIR}',
    local_dir_use_symlinks=False,
)
print(f'DOWNLOADED:{path}')
`;

  const tmpScript = path.join(DATA_DIR, '_download.py');
  fs.writeFileSync(tmpScript, pythonScript);

  try {
    const result = execSync(`HF_TOKEN=${hfToken} python3 ${tmpScript}`, {
      encoding: 'utf-8',
      timeout: 300000,
    });
    log('Download result:', result.trim());
  } catch (error) {
    throw new Error(`Dataset download failed: ${error.message}`);
  } finally {
    fs.unlinkSync(tmpScript);
  }

  // Convert parquet to JSON
  const convertScript = `
import json
import pyarrow.parquet as pq
import os

parquet_path = os.path.join('${DATA_DIR}', 'data', 'train.parquet')
output_path = '${JSON_PATH}'

# Also check direct path
if not os.path.exists(parquet_path):
    parquet_path = os.path.join('${DATA_DIR}', 'train.parquet')

if not os.path.exists(parquet_path):
    # Search for it
    import glob
    files = glob.glob('${DATA_DIR}/**/*.parquet', recursive=True)
    if files:
        parquet_path = files[0]

print(f'Reading parquet from: {parquet_path}')
table = pq.read_table(parquet_path)
df = table.to_pandas()

with open(output_path, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict(), ensure_ascii=False) + '\\n')

print(f'CONVERTED:{len(df)} rows to {output_path}')
`;

    const convertScriptPath = path.join(DATA_DIR, '_convert.py');
    fs.writeFileSync(convertScriptPath, convertScript);

  try {
    const result = execSync(`python3 ${convertScriptPath}`, {
      encoding: 'utf-8',
      timeout: 120000,
    });
    log('Convert result:', result.trim());
  } catch (error) {
    throw new Error(`Parquet conversion failed: ${error.message}`);
  } finally {
    fs.unlinkSync(convertScriptPath);
  }
}

async function loadDatasetFromJSON() {
  if (!fs.existsSync(JSON_PATH)) {
    throw new Error(`Dataset JSON not found at ${JSON_PATH}. Run download first.`);
  }

  const content = fs.readFileSync(JSON_PATH, 'utf-8');
  const lines = content.split('\n').filter(Boolean);
  const rows = lines.map((line, index) => {
    try {
      return { row: JSON.parse(line) };
    } catch (e) {
      warn(`Failed to parse line ${index}: ${e.message}`);
      return null;
    }
  }).filter(Boolean);

  return { rows, total: rows.length };
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

  // Step 1: Download dataset if needed
  await downloadDataset();

  // Step 2: Load from JSON
  const { rows: allRows, total: totalRows } = await loadDatasetFromJSON();
  log(`Loaded ${totalRows} recipes from JSON.`);

  // Step 3: Connect to ChromaDB
  const collection = await getOrCreateCollection();
  const existingCount = await getCollectionCount(collection);

  if (existingCount > 0) {
    log(`Collection already contains ${existingCount} documents. Skipping ingestion.`);
    return { skipped: true, count: existingCount };
  }

  // Step 4: Test batch
  log(`Running small-batch ingestion test with ${testBatchSize} recipes.`);
  const testRows = allRows.slice(0, testBatchSize);
  const testRecipes = testRows.map((row, index) => buildRecipeDocument(row.row, index));
  await addRecipesToCollection(collection, testRecipes);
  let ingestedCount = testRecipes.length;
  log(`Ingested ${ingestedCount}/${totalRows} recipes (test batch).`);

  // Step 5: Full ingestion
  for (let offset = ingestedCount; offset < totalRows; offset += batchSize) {
    const currentBatchSize = Math.min(batchSize, totalRows - offset);
    const batchRows = allRows.slice(offset, offset + currentBatchSize);
    const recipes = batchRows.map((row, index) => buildRecipeDocument(row.row, offset + index));

    if (recipes.length === 0) break;

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
  downloadDataset,
};

if (require.main === module) {
  ingestRecipes().catch((error) => {
    console.error('[ingest] Fatal error during ingestion:', error);
    process.exit(1);
  });
}
