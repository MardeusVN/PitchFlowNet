"""Command-line entry point for Banhmi-TTS preprocessing."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .config import AudioConfig, PreprocessConfig, SplitConfig, TextConfig
from .pipeline import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(prog="python -m banhmi_tts.preprocessing")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="Optional preprocessing JSON")
    parser.add_argument("--sample-rate", type=int, default=22_050)
    parser.add_argument("--language", default="en-us", choices=("en", "en-us", "en-gb"))
    parser.add_argument("--num-validation", type=int, default=100)
    parser.add_argument("--num-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1_234)
    parser.add_argument("--max-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, help="Process only the first N records")
    parser.add_argument("--force", action="store_true", help="Ignore valid cache records")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    if args.config is not None:
        config_value = json.loads(args.config.read_text(encoding="utf-8"))
        # Accept either a standalone preprocessing config or config.json from
        # a previous output directory.
        config = PreprocessConfig.from_dict(
            config_value.get("preprocessing", config_value)
        )
    else:
        config = PreprocessConfig(
            audio=AudioConfig(sample_rate=args.sample_rate),
            text=TextConfig(language=args.language),
            split=SplitConfig(
                num_validation=args.num_validation,
                num_test=args.num_test,
                seed=args.seed,
            ),
        )

    report = prepare_dataset(
        args.input_dir,
        args.output_dir,
        config,
        max_workers=args.max_workers,
        force=args.force,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
