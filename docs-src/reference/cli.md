# CLI

Three subcommands, each with a matching Python entry point. `ecgbench/cli/_main.py`
builds the root parser and dispatches on `args.func`; every subcommand module
exposes a public `run_X(...)` taking plain keyword arguments, so anything the CLI
can do is importable — see [Pipelines](api/pipelines.md) for their generated
signatures.

--8<-- "README.md:cli"
