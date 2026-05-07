#!/bin/bash
# Send 100 queries and report P50/P95/P99 latency

echo "Load testing RAG API (100 queries)..."

questions=(
  "What is the main cause of climate change?"
  "How does solar energy work?"
  "What are greenhouse gas emissions?"
)

total=100
for i in $(seq 1 $total); do
  q="${questions[$((i % 3))]}"

  start=$(date +%s%N)
  curl -s -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d "{\"question\": \"$q\", \"top_k\": 5}" > /dev/null
  end=$(date +%s%N)

  latency=$(( (end - start) / 1000000 ))
  echo "$latency"
done | awk '{
  sum += $1
  arr[NR] = $1
} END {
  asort(arr)
  n = NR
  p50 = arr[int(n * 0.50)]
  p95 = arr[int(n * 0.95)]
  p99 = arr[int(n * 0.99)]
  printf "Requests: %d\n", n
  printf "P50: %d ms\n", p50
  printf "P95: %d ms\n", p95
  printf "P99: %d ms\n", p99
  printf "Mean: %d ms\n", sum / n
}'
