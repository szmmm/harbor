"""Create Harbor tasks from arbitrary prompt JSONL."""

import argparse
from pathlib import Path

from .adapter import PromptAdapter, PromptDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", type=Path, required=True)
    parser.add_argument("--prompt-field", default="query")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset = PromptDataset(args.jsonl, args.prompt_field)
    generated = PromptAdapter(
        args.output_dir,
        dataset=dataset,
        limit=args.limit,
        overwrite=args.overwrite,
    ).run()
    print(f"Generated {len(generated)} tasks in {args.output_dir}")


if __name__ == "__main__":
    main()
