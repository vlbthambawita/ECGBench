"""Full pipeline subcommand: validate + split + export + Croissant."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def run_splits(
    dataset: str,
    data_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    sampling_rate: int | None = None,
    n_folds: int | None = None,
    max_workers: int = 4,
    skip_validation: bool = False,
    skip_croissant: bool = False,
) -> dict:
    """Run the full pipeline: validate + split + export + Croissant.

    ``n_folds`` defaults to the dataset's own ``config.n_folds``, which is 10 for
    every dataset but ``szdb`` — see that field for why a small release cannot
    take ten. Passing an explicit value overrides it.

    Returns the stats dict produced by ``export_splits`` plus ``output_dir``,
    ``dataset`` and ``dataset_name`` keys for convenience.
    """
    from ecgbench.config import load_config
    from ecgbench.download import resolve_data_path
    from ecgbench.splitting import export_splits, get_splitter, split_dataset

    logger.info("Loading config for '%s'", dataset)
    config = load_config(dataset)
    if n_folds is None:
        n_folds = config.n_folds

    resolved_data_path = resolve_data_path(Path(data_path) if data_path else None, config)
    logger.info("Data path: %s", resolved_data_path)

    splitter = get_splitter(dataset)
    logger.info("Using splitter: %s", type(splitter).__name__)

    df = splitter.load_metadata(resolved_data_path, config)
    labels = splitter.get_stratification_labels(df, config)

    if not skip_validation:
        from ecgbench.validation import validate_dataset

        logger.info("Validating dataset...")
        val_result = validate_dataset(
            resolved_data_path,
            config,
            sampling_rate=sampling_rate,
            max_workers=max_workers,
        )
        logger.info(
            "Validation: %d total, %d valid, %d excluded",
            val_result.total_records,
            val_result.valid_records,
            val_result.excluded_records,
        )
    else:
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

    logger.info("Splitting dataset into %d folds...", n_folds)
    split_result = split_dataset(df, labels, config, n_folds=n_folds)
    logger.info(
        "Split complete: %d folds, train=%s, val=%s, test=%s",
        split_result.n_folds,
        split_result.default_train_folds,
        split_result.default_val_folds,
        split_result.default_test_folds,
    )

    resolved_output = Path(output_dir) if output_dir else Path("output") / dataset
    logger.info("Exporting to %s", resolved_output)
    stats = export_splits(split_result, val_result, resolved_output, config)

    manifest = _write_manifest(config, resolved_data_path, resolved_output, split_result, n_folds)

    if not skip_croissant:
        try:
            from ecgbench.croissant import save_croissant

            for version in ("clean", "original"):
                version_dir = resolved_output / version
                if version_dir.exists():
                    croissant_path = version_dir / "croissant.json"
                    save_croissant(config, version_dir, croissant_path, version=version)
                    logger.info("Generated Croissant metadata: %s", croissant_path)
        except ImportError:
            logger.warning(
                "mlcroissant not installed — skipping Croissant generation. "
                "Install with: pip install ecgbench[croissant]"
            )

    return {
        "dataset": config.slug,
        "dataset_name": config.name,
        "output_dir": resolved_output,
        "manifest": manifest,
        **stats,
    }


def _write_manifest(config, data_path, output_dir, split_result, n_folds: int) -> dict:
    """Write manifest.json describing how this split was produced.

    Emitted for every dataset, not only the unpublishable ones: the fold digest
    is the cheapest way for anyone to confirm two runs produced the same
    partition, and the input checksums say what "the same" was computed from.
    """
    import json

    import pandas as pd

    from ecgbench.manifest import build_manifest

    # Read back what export_splits actually wrote rather than recomputing the
    # is_valid merge, which lives inside it. The digest then describes the
    # published artefact, not an approximation of it.
    original = pd.read_csv(Path(output_dir) / "original" / "folds.csv")
    clean = pd.read_csv(Path(output_dir) / "clean" / "folds.csv")

    manifest = build_manifest(
        config=config,
        data_path=Path(data_path),
        original_df=original,
        clean_df=clean,
        n_folds=n_folds,
        random_state=split_result.split_metadata.get("random_state"),
    )
    path = Path(output_dir) / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "Wrote %s (fold digest clean=%s)",
        path,
        manifest["fold_digest"]["clean"][:16] + "...",
    )
    return manifest


def _print_summary(result: dict) -> None:
    print("\n" + "=" * 60)
    print(f"  Dataset: {result['dataset_name']} ({result['dataset']})")
    print(f"  Output:  {result['output_dir']}")
    print("=" * 60)
    print(f"  Original: {result['original']['total']} records")
    print(f"    Train: {result['original']['train']}")
    print(f"    Val:   {result['original']['val']}")
    print(f"    Test:  {result['original']['test']}")
    print(f"  Clean:    {result['clean']['total']} records")
    print(f"    Train: {result['clean']['train']}")
    print(f"    Val:   {result['clean']['val']}")
    print(f"    Test:  {result['clean']['test']}")
    print(f"  Excluded: {result['excluded']}")
    print("=" * 60)


def _cli_run(args: argparse.Namespace) -> int:
    result = run_splits(
        dataset=args.dataset,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sampling_rate=args.sampling_rate,
        n_folds=args.n_folds,
        max_workers=args.max_workers,
        skip_validation=args.skip_validation,
        skip_croissant=args.skip_croissant,
    )
    _print_summary(result)
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "splits",
        help="Validate + split + export + Croissant metadata for a dataset",
        description="Full pipeline: validate + split + export + Croissant generation.",
    )
    p.add_argument(
        "--dataset",
        required=True,
        help="Dataset slug (e.g., 'ptbxl', 'chapman_shaoxing')",
    )
    p.add_argument(
        "--data-path",
        default=None,
        help="Path to dataset root directory. If omitted, auto-downloads.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Defaults to output/<dataset>/",
    )
    p.add_argument(
        "--sampling-rate",
        type=int,
        default=None,
        help="Sampling rate to validate (default: dataset's default_sampling_rate)",
    )
    p.add_argument(
        "--n-folds",
        type=int,
        default=None,
        help=(
            "Number of folds. Defaults to the dataset's configured n_folds, "
            "which is 10 everywhere except szdb (5 — it has 7 records from 5 "
            "subjects, and ten folds is arithmetically impossible there)."
        ),
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel validation workers (default: 4)",
    )
    p.add_argument(
        "--skip-validation",
        action="store_true",
        help="Skip signal validation (faster, but no quality flags)",
    )
    p.add_argument(
        "--skip-croissant",
        action="store_true",
        help="Skip Croissant metadata generation",
    )
    p.set_defaults(func=_cli_run)
    return p
