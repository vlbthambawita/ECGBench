"""The MCP server (metadata layer, Phase 7): tool functions, the built server,
and one real stdio round trip against a spawned ``ecgbench mcp``.

The tool functions need only the store, so the first class runs without the
``mcp`` SDK; the rest ``importorskip`` it. The spawned-server test is the
automated form of the plan's manual check ("ask the agent to list 2-lead
Holter datasets").
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from ecgbench.cli import main
from ecgbench.metadata.mcp_server import (
    INSTRUCTIONS,
    SERVER_NAME,
    TOOL_NAMES,
    ToolFunctions,
    summary,
)

# --------------------------------------------------------------------------- tool functions


class TestToolFunctions:
    @pytest.fixture
    def tools(self):
        return ToolFunctions()

    def test_search_is_ranked_and_summarised(self, tools):
        hits = tools.search_datasets("holter", leads=2, limit=3)
        assert len(hits) == 3
        assert all(h["leads"] == 2 for h in hits)
        assert set(hits[0]) >= {
            "dataset_id", "name", "implementation_state", "records", "leads",
            "sampling_rates", "access", "license", "has_labels", "fold_csvs_published",
            "description", "score",
        }
        json.dumps(hits)  # JSON-ready, no tuples or dataclasses left

    def test_filters_without_a_query(self, tools):
        hits = tools.search_datasets(access="credentialed", limit=50)
        assert hits and all(h["access"] == "credentialed" for h in hits)
        assert "mimic_iv_ecg" in {h["dataset_id"] for h in hits}

    def test_list_filters_by_state_and_category(self, tools):
        everything = tools.list_datasets()
        assert len(everything) == 64
        published = tools.list_datasets(state="published")
        assert published and all(m["implementation_state"] == "published" for m in published)
        two_lead = tools.list_datasets(category="two-lead")
        assert {m["dataset_id"] for m in two_lead} >= {"mitdb", "afdb"}

    def test_get_resolves_any_alias_and_is_json_ready(self, tools):
        record = tools.get_dataset("PTB-XL")
        assert record["dataset_id"] == "ptbxl"
        assert isinstance(record["aliases"], list)  # to_dict holds tuples; MCP wants lists
        assert record == tools.get_dataset("ptb-xl") == tools.get_dataset("ptbxl")
        json.dumps(record)

    def test_fields_and_relations(self, tools):
        fields = tools.list_fields("mitdb")
        assert {f["name"] for f in fields} >= {"age", "recorder"}
        assert fields[0].keys() >= {"name", "type", "description", "unit", "vocabulary"}
        assert tools.list_fields("mimic_iv_ecg_demo") == []
        related = tools.related_datasets("ptb-xl")
        assert related and all(r["source"] == "ptbxl" for r in related)
        assert {"target", "relation", "shares_records", "note"} <= set(related[0])

    def test_errors_carry_a_message(self, tools):
        with pytest.raises(Exception, match="unknown dataset 'nope'"):
            tools.get_dataset("nope")
        with pytest.raises(Exception, match="FTS5 syntax"):
            tools.search_datasets("ptb-xl")
        with pytest.raises(Exception, match="state must be one of"):
            tools.list_datasets(state="bogus")
        with pytest.raises(Exception, match="state must be one of"):
            tools.search_datasets("ecg", state="bogus")

    def test_summary_of_a_catalogue_only_dataset(self):
        from ecgbench.metadata import get

        row = summary(get("ptb-xl-plus"))
        assert row["implementation_state"] == "catalogue_only"
        assert row["has_labels"] is False and row["signal_format"] is None


# --------------------------------------------------------------------------- built server


def _error(result):
    return result.isError if hasattr(result, "isError") else result.is_error


def _structured(result):
    if hasattr(result, "structuredContent"):
        return result.structuredContent
    return result.structured_content


class TestBuiltServer:
    @pytest.fixture
    def server(self):
        pytest.importorskip("mcp")
        from ecgbench.metadata.mcp_server import build_mcp_server

        return build_mcp_server()

    def test_registers_the_five_tools_with_schemas(self, server):
        tools = asyncio.run(server.list_tools())
        assert [t.name for t in tools] == list(TOOL_NAMES)
        by_name = {t.name: t for t in tools}
        schema = getattr(by_name["get_dataset"], "input_schema", None) or getattr(
            by_name["get_dataset"], "inputSchema"
        )
        assert schema["required"] == ["dataset"]
        assert "FTS5" in by_name["search_datasets"].description
        assert server.name == SERVER_NAME and server.instructions == INSTRUCTIONS

    def test_call_tool_returns_structured_content(self, server):
        result = asyncio.run(server.call_tool("search_datasets", {"query": "holter", "leads": 2}))
        assert not _error(result)
        ids = [h["dataset_id"] for h in _structured(result)["result"]]
        assert "mitdb" in ids and "afdb" in ids

    def test_tool_errors_keep_their_message(self, server):
        with pytest.raises(Exception, match="unknown dataset 'nope'"):
            asyncio.run(server.call_tool("get_dataset", {"dataset": "nope"}))


# --------------------------------------------------------------------------- over the wire


class TestStdioRoundTrip:
    """Spawn ``ecgbench mcp`` and talk to it as a client would."""

    def test_list_two_lead_holter_datasets(self):
        pytest.importorskip("mcp")
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=sys.executable,
            args=["-c", "from ecgbench.cli import main; raise SystemExit(main(['mcp']))"],
        )

        async def session_run():
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    init = await session.initialize()
                    info = getattr(init, "server_info", None) or init.serverInfo
                    listed = await session.list_tools()
                    hits = await session.call_tool(
                        "search_datasets", {"query": "holter", "leads": 2, "limit": 5}
                    )
                    unknown = await session.call_tool("get_dataset", {"dataset": "nope"})
                    return info.name, [t.name for t in listed.tools], hits, unknown

        name, tool_names, hits, unknown = asyncio.run(session_run())
        assert name == SERVER_NAME
        assert tool_names == list(TOOL_NAMES)
        assert not _error(hits)
        rows = _structured(hits)["result"]
        assert len(rows) == 5 and all(r["leads"] == 2 for r in rows)
        assert _error(unknown) and "unknown dataset 'nope'" in unknown.content[0].text


# --------------------------------------------------------------------------- CLI


def test_cli_rejects_a_bad_transport_and_reports_a_missing_sdk(monkeypatch, capsys):
    from ecgbench.cli.mcp import run_mcp

    with pytest.raises(ValueError, match="transport"):
        run_mcp("carrier-pigeon")
    with pytest.raises(SystemExit):
        main(["mcp", "--transport", "carrier-pigeon"])

    import ecgbench.metadata.mcp_server as mod

    def missing(store=None):
        raise ImportError("the MCP server needs the mcp SDK: pip install ecgbench[mcp]")

    monkeypatch.setattr(mod, "build_mcp_server", missing)
    assert main(["mcp"]) == 1
    assert "ecgbench[mcp]" in capsys.readouterr().err
