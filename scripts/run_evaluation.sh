#!/bin/bash
set -e

echo "Running evaluation on golden dataset..."

response_file=$(mktemp)
status=$(curl -sS -o "$response_file" -w "%{http_code}" http://localhost:8000/evaluate)
response=$(cat "$response_file")
rm -f "$response_file"

if [ "$status" -ne 200 ]; then
  echo "Evaluation endpoint returned HTTP $status"
  echo "$response"
  exit 1
fi

if ! echo "$response" | jq '.'; then
  echo "Evaluation endpoint returned non-JSON response:"
  echo "$response"
  exit 1
fi


passed=$(echo "$response" | jq '.passed')

if [ "$passed" = "true" ]; then
  echo "Evaluation PASSED"
  exit 0
else
  echo "Evaluation FAILED"
  exit 1
fi
