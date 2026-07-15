"""MCP stdio integration smoke test against the running Shield API."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


BACKEND_DIR = Path(__file__).resolve().parents[1]


def result_payload(result) -> dict:
    if result.structuredContent:
        return result.structuredContent
    return json.loads(result.content[0].text)


async def smoke() -> None:
    environment = os.environ.copy()
    environment.setdefault("SHIELD_API_BASE_URL", "http://localhost:8000")
    parameters = StdioServerParameters(
        command=sys.executable,
        args=[str(BACKEND_DIR / "mcp_server.py")],
        env=environment,
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            tools = await session.list_tools()
            names = {tool.name for tool in tools.tools}
            assert {
                "shield_health",
                "analyze_fraud_text",
                "queue_fraud_text",
                "get_analysis_job",
                "list_recent_cases",
                "get_case_evidence",
                "hotspot_overview",
                "fraud_network_summary",
                "reporting_guidance",
            }.issubset(names)
            tool_map = {tool.name: tool for tool in tools.tools}
            assert tool_map["shield_health"].annotations.readOnlyHint is True
            assert tool_map["analyze_fraud_text"].annotations.readOnlyHint is False
            assert tool_map["analyze_fraud_text"].annotations.destructiveHint is False
            resources = await session.list_resources()
            resource_uris = {str(resource.uri) for resource in resources.resources}
            assert "shield://capabilities" in resource_uris
            assert "shield://reporting-guidance" in resource_uris
            prompts = await session.list_prompts()
            assert "fraud_triage" in {prompt.name for prompt in prompts.prompts}
            result = await session.call_tool("shield_health")
            assert not result.isError
            capability = await session.read_resource("shield://capabilities")
            assert capability.contents
            hotspot = await session.call_tool("hotspot_overview")
            assert not hotspot.isError
            prompt = await session.get_prompt("fraud_triage", {"language": "hi"})
            assert prompt.messages
            if os.getenv("SHIELD_API_TOKEN"):
                queued = await session.call_tool(
                    "queue_fraud_text",
                    {
                        "text": "Share your OTP immediately or your bank account will be blocked.",
                        "language": "en",
                    },
                )
                assert not queued.isError
                job_id = result_payload(queued)["job_id"]
                for _ in range(90):
                    status_result = await session.call_tool(
                        "get_analysis_job", {"job_id": job_id}
                    )
                    assert not status_result.isError
                    job = result_payload(status_result)
                    if job["status"] in {"completed", "failed"}:
                        break
                    await asyncio.sleep(1)
                assert job["status"] == "completed"
                assert job["result"]["response_language"] == "en"

    print("MCP tools, resources, prompt, health, hotspot, and optional queue flow: PASS")


if __name__ == "__main__":
    asyncio.run(smoke())
