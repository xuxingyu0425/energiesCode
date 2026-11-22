"""Score test predictions at a previously frozen validation operating point."""

import argparse
import json
from pathlib import Path

from common import counts, load_json, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-cache", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("--output must point to a new file.")

    test = load_json(args.test_cache)
    selection = load_json(args.selection)
    if test.get("split") != "test" or selection.get("test_data_used") is not False:
        parser.error("Use a test cache and an operating point selected on validation data.")
    if test.get("weight_sha256") != selection.get("weight_sha256") or not test.get("weight_sha256"):
        parser.error("The test cache and validation operating point must use the same checkpoint.")

    threshold = selection["threshold"]
    ground_truth, predictions, true_positives = counts(test["records"], threshold)
    result = {"threshold": threshold, "weight_sha256": test["weight_sha256"],
              **metrics(ground_truth.sum(), predictions.sum(), true_positives.sum())}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
