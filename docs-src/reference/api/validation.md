# Validation

Every record is checked before it is split, producing the `original` version
(all records, plus `is_valid` and `quality_issues`) and `clean` (valid only).

## Engine

::: ecgbench.validation.engine
    options:
      members:
        - validate_dataset
        - ValidationResult

## Checks

Each check is a function over a signal array, registered in `CHECK_REGISTRY` and
selected per dataset by the config's `validation:` block.

::: ecgbench.validation.checks

## Report

::: ecgbench.validation.report
