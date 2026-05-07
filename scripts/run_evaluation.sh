#!/bin/bash
set -e

echo "Running evaluation on golden dataset..."

response=$(curl -s http://localhost:8000/evaluate)

echo "$response" | jq '.'

passed=$(echo "$response" | jq '.passed')

if [ "$passed" = "true" ]; then
  echo "Evaluation PASSED"
  exit 0
else
  echo "Evaluation FAILED"
  exit 1
fi
