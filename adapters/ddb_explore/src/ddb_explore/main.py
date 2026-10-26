"""Create Harbor tasks from a rubrics-tree-generation rubrics.jsonl."""

import argparse
import json
from pathlib import Path

from .adapter import RubricAdapter, RubricDataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--prompt-data",
        type=Path,
        help="Also write the slime prompt data JSONL here (one row per task).",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    dataset = RubricDataset(args.rubrics)
    adapter = RubricAdapter(
        args.output_dir,
        dataset=dataset,
        limit=args.limit,
        overwrite=args.overwrite,
    )
    generated = adapter.run()
    print(f"Generated {len(generated)} tasks in {args.output_dir}")
    if args.prompt_data is not None:
        task_ids = dataset.ids()[: args.limit] if args.limit is not None else dataset.ids()
        rows = adapter.prompt_rows(task_ids)
        args.prompt_data.parent.mkdir(parents=True, exist_ok=True)
        args.prompt_data.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        print(f"Wrote {len(rows)} prompt rows to {args.prompt_data}")


if __name__ == "__main__":
    main()
