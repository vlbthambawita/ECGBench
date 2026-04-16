#!/usr/bin/env python3
"""
Full pipeline: validate + split + export + Croissant generation.

Usage:
    python scripts/generate_splits.py --dataset ptbxl --data-path /path/to/ptb-xl/1.0.3/
    python scripts/generate_splits.py --dataset ptbxl  # auto-download
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Generate validated fold splits and Croissant metadata for an ECG dataset"
    )
    parser.add_argument(
        "--dataset", required=True,
        help="Dataset slug (e.g., 'ptbxl', 'chapman_shaoxing')",
    )
    parser.add_argument(
        "--data-path", default=None,
        help="Path to dataset root directory. If omitted, auto-downloads.",
    )
    parser.add_argument(
        "--output-dir", default=None,
        help="Output directory. Defaults to output/<dataset>/",
    )
    parser.add_argument(
        "--sampling-rate", type=int, default=None,
        help="Sampling rate to validate (default: dataset's default_sampling_rate)",
    )
    parser.add_argument(
        "--n-folds", type=int, default=10,
        help="Number of folds (default: 10)",
    )
    parser.add_argument(
        "--max-workers", type=int, default=4,
        help="Number of parallel validation workers (default: 4)",
    )
    parser.add_argument(
        "--skip-validation", action="store_true",
        help="Skip signal validation (faster, but no quality flags)",
    )
    parser.add_argument(
        "--skip-croissant", action="store_true",
        help="Skip Croissant metadata generation",
    )
    args = parser.parse_args()

    from pathlib import Path

    from ecgbench.config import load_config
    from ecgbench.download import resolve_data_path
    from ecgbench.splitting import export_splits, get_splitter, split_dataset

    # 1. Load config
    logger.info("Loading config for '%s'", args.dataset)
    config = load_config(args.dataset)

    # 2. Resolve data path
    data_path = resolve_data_path(
        Path(args.data_path) if args.data_path else None, config
    )
    logger.info("Data path: %s", data_path)

    # 3. Get splitter and load metadata
    splitter = get_splitter(args.dataset)
    logger.info("Using splitter: %s", type(splitter).__name__)

    df = splitter.load_metadata(data_path, config)
    labels = splitter.get_stratification_labels(df, config)

    # 4. Validate
    if not args.skip_validation:
        from ecgbench.validation import validate_dataset

        logger.info("Validating dataset...")
        val_result = validate_dataset(
            data_path, config,
            sampling_rate=args.sampling_rate,
            max_workers=args.max_workers,
        )
        logger.info(
            "Validation: %d total, %d valid, %d excluded",
            val_result.total_records, val_result.valid_records, val_result.excluded_records,
        )
    else:
        # Create a dummy validation result where everything is valid
        from ecgbench.validation.engine import ValidationResult

        original_df = df.copy()
        original_df["is_valid"] = True
        original_df["quality_issues"] = ""
        val_result = ValidationResult(
            original_df=original_df,
            clean_df=df.copy(),
            record_validations=[],
            summary={},
            total_records=len(df),
            valid_records=len(df),
            excluded_records=0,
        )
        logger.info("Validation skipped — all %d records marked as valid", len(df))

    # 5. Split
    logger.info("Splitting dataset into %d folds...", args.n_folds)
    split_result = split_dataset(df, labels, config, n_folds=args.n_folds)
    logger.info(
        "Split complete: %d folds, train=%s, val=%s, test=%s",
        split_result.n_folds,
        split_result.default_train_folds,
        split_result.default_val_folds,
        split_result.default_test_folds,
    )

    # 6. Export
    output_dir = Path(args.output_dir) if args.output_dir else Path("output") / args.dataset
    logger.info("Exporting to %s", output_dir)
    stats = export_splits(split_result, val_result, output_dir, config)

    # 7. Croissant
    if not args.skip_croissant:
        try:
            from ecgbench.croissant import save_croissant

            for version in ("clean", "original"):
                version_dir = output_dir / version
                if version_dir.exists():
                    save_croissant(config, version_dir, version=version)
                    logger.info("Generated Croissant metadata for %s version", version)
        except ImportError:
            logger.warning(
                "mlcroissant not installed — skipping Croissant generation. "
                "Install with: pip install ecgbench[croissant]"
            )

    # 8. Print summary
    print("\n" + "=" * 60)
    print(f"  Dataset: {config.name} ({config.slug})")
    print(f"  Output:  {output_dir}")
    print("=" * 60)
    print(f"  Original: {stats['original']['total']} records")
    print(f"    Train: {stats['original']['train']}")
    print(f"    Val:   {stats['original']['val']}")
    print(f"    Test:  {stats['original']['test']}")
    print(f"  Clean:    {stats['clean']['total']} records")
    print(f"    Train: {stats['clean']['train']}")
    print(f"    Val:   {stats['clean']['val']}")
    print(f"    Test:  {stats['clean']['test']}")
    print(f"  Excluded: {stats['excluded']}")
    print("=" * 60)


if __name__ == "__main__":
    main()
