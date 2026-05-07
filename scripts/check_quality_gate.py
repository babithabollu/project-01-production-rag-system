"""CI quality gate: fails with exit code 1 if evaluation metrics are below threshold."""
import sys

import requests

response = requests.get("http://localhost:8000/evaluate", timeout=300)
response.raise_for_status()
result = response.json()

if not result["passed"]:
    print("Quality gate FAILED")
    print(
        f"Faithfulness: {result['metrics']['faithfulness']:.3f} "
        f"(threshold: {result['thresholds']['min_faithfulness']})"
    )
    print(
        f"Precision@5:  {result['metrics']['context_precision@5']:.3f} "
        f"(threshold: {result['thresholds']['min_precision_at_5']})"
    )
    sys.exit(1)
else:
    print("Quality gate PASSED")
    print(f"Faithfulness: {result['metrics']['faithfulness']:.3f}")
    print(f"Precision@5:  {result['metrics']['context_precision@5']:.3f}")
    sys.exit(0)
