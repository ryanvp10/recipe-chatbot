FROM node:22-slim

WORKDIR /app

# Install Python and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy package files
COPY package.json package-lock.json ./
RUN npm ci --only=production

# Copy source code
COPY src/ src/

# Copy embeddings data
COPY embeddings_data/ embeddings_data/

# HuggingFace Spaces uses port 7860
ENV PORT=7860
ENV HF_TOKEN=your_hf_token_here
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD node -e "const http = require('http'); http.get('http://localhost:7860/api/health', (r) => { process.exit(r.statusCode === 200 ? 0 : 1); }).on('error', () => process.exit(1));"

# Start server
CMD ["node", "src/server.js"]
