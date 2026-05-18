# Task: Run Dataset Ingestion

The dataset is already downloaded and converted to JSON at:
/home/ubuntu/recipe-chatbot/backend/data/train.json (66,419 recipes)

ChromaDB server should already be running on localhost:8000. If not, start it:
```
/home/ubuntu/.hermes/profiles/worker/home/.local/bin/chroma run --path /home/ubuntu/recipe-chatbot/backend/chroma_data --port 8000 > /tmp/chroma.log 2>&1 &
```

Run the ingestion script:
```
cd /home/ubuntu/recipe-chatbot/backend && node src/ingest.js
```

This will take time — it needs to embed 66,419 recipes. Wait for it to complete.

If there are errors, read the file, fix the code, and re-run.

After successful ingestion, commit:
```
cd /home/ubuntu/recipe-chatbot && git add -A && git commit -m "feat: dataset ingestion complete"
```

Report the final output.
