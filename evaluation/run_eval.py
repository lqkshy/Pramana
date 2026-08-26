"""
CLI entry-point for running Pramana evaluations.

Usage:
    python -m evaluation.run_eval --dataset datasets/fever_sample.jsonl

Loads a labelled dataset, runs the pipeline against each example,
and prints a metrics report to stdout.
"""
import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Run Pramana evals")
    parser.add_argument("--dataset", required=True, help="Path to JSONL eval dataset")
    args = parser.parse_args()
    # TODO: implement eval loop
    print(f"Running eval on: {args.dataset}")

if __name__ == "__main__":
    main()
