"""``ecgbench mcp [--transport stdio|sse|streamable-http]``.

Serves the metadata layer to agents over the Model Context Protocol: five
tools (``search_datasets``, ``list_datasets``, ``get_dataset``, ``list_fields``,
``related_datasets``) that read the bundled index and nothing else. Register
it with Claude Code as ``claude mcp add ecgbench -- ecgbench mcp``.
"""

from __future__ import annotations

import argparse
import sys

TRANSPORTS = ("stdio", "sse", "streamable-http")


def run_mcp(transport: str = "stdio") -> None:
    """Serve the metadata tools until the client disconnects.

    Args:
        transport: ``stdio`` (what MCP clients spawn and the default), ``sse`` or
            ``streamable-http``.

    Raises:
        ImportError: the ``mcp`` SDK is not installed (``pip install ecgbench[mcp]``).
        ValueError: unknown transport.
    """
    if transport not in TRANSPORTS:
        raise ValueError(f"transport must be one of {TRANSPORTS}, got {transport!r}")
    from ecgbench.metadata.mcp_server import run_server

    run_server(transport)


def _cli_run(args: argparse.Namespace) -> int:
    try:
        run_mcp(args.transport)
    except ImportError as exc:
        print(f"ecgbench: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 0
    return 0


def add_subparser(subparsers) -> argparse.ArgumentParser:
    p = subparsers.add_parser(
        "mcp",
        help="Serve the metadata layer to agents over the Model Context Protocol",
        description=(
            "An MCP server with five tools over the bundled metadata index: search_datasets, "
            "list_datasets, get_dataset, list_fields and related_datasets. Reads no records, "
            "signals or network. Needs the mcp extra: pip install ecgbench[mcp]."
        ),
        epilog="register with Claude Code:  claude mcp add ecgbench -- ecgbench mcp",
    )
    p.add_argument(
        "--transport",
        choices=TRANSPORTS,
        default="stdio",
        help="Transport to serve on (default: stdio, which MCP clients spawn)",
    )
    p.set_defaults(func=_cli_run)
    return p
