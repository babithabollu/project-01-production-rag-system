#!/bin/bash
set -e

echo "Ingesting demo documents..."

for file in data/documents/*; do
  echo "  - Uploading: $(basename "$file")"
  curl -X POST http://localhost:8000/ingest \
    -F "file=@$file" \
    -s | jq '.status'
done

echo "Ingestion complete"
