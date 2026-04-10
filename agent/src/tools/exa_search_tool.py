"""Exa Search tool: AI-powered web search with content extraction.

Exa finds relevant pages semantically (not just keyword matching).
Returns URLs + extracted content for any query.

API: https://api.exa.ai/search
Auth: x-api-key header
"""

from __future__ import annotations

import json
import os
from typing import Any

import requests

from src.agent.tools import BaseTool

EXA_API_KEY = os.environ.get("EXA_API_KEY", "")
EXA_BASE = "https://api.exa.ai"
_TIMEOUT = 15


def _exa_request(endpoint: str, body: dict) -> dict:
    """Make authenticated Exa API request."""
    if not EXA_API_KEY:
        return {"error": "EXA_API_KEY not configured"}
    try:
        resp = requests.post(
            f"{EXA_BASE}{endpoint}",
            json=body,
            headers={
                "x-api-key": EXA_API_KEY,
                "Content-Type": "application/json",
            },
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        return {"error": str(e)}


class ExaSearchTool(BaseTool):
    """Search the web semantically using Exa AI."""

    name = "exa_search"
    description = (
        "Search the web using AI-powered semantic search. "
        "Better than Google for finding specific information — "
        "understands meaning, not just keywords. "
        "Use for: news, research papers, company info, market analysis, "
        "insider trading reports, earnings analysis, macro data. "
        "Returns URLs with extracted text content."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query in natural language, e.g. 'AAPL insider trading activity 2026' or 'BTC whale accumulation on-chain data'",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default 5, max 10)",
            },
            "type": {
                "type": "string",
                "description": "Search type: 'auto' (default), 'neural' (semantic), 'keyword' (traditional)",
                "enum": ["auto", "neural", "keyword"],
            },
        },
        "required": ["query"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        query = kwargs["query"]
        num = min(kwargs.get("num_results", 5), 10)
        search_type = kwargs.get("type", "auto")

        data = _exa_request("/search", {
            "query": query,
            "numResults": num,
            "type": search_type,
            "contents": {
                "text": {"maxCharacters": 2000},
            },
        })

        if "error" in data:
            return json.dumps(data, ensure_ascii=False)

        results = data.get("results", [])
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "published": r.get("publishedDate", ""),
                "score": round(r.get("score", 0), 3),
                "text": (r.get("text", "") or "")[:1500],
            })

        return json.dumps({
            "status": "ok",
            "query": query,
            "count": len(formatted),
            "results": formatted,
        }, ensure_ascii=False)


class ExaFindSimilarTool(BaseTool):
    """Find pages similar to a given URL."""

    name = "exa_similar"
    description = (
        "Find web pages similar to a given URL. "
        "Useful for finding related research, competitor analysis, "
        "or more sources on the same topic."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to find similar pages for",
            },
            "num_results": {
                "type": "integer",
                "description": "Number of results (default 5)",
            },
        },
        "required": ["url"],
    }
    repeatable = True

    def execute(self, **kwargs: Any) -> str:
        url = kwargs["url"]
        num = min(kwargs.get("num_results", 5), 10)

        data = _exa_request("/findSimilar", {
            "url": url,
            "numResults": num,
            "contents": {
                "text": {"maxCharacters": 1500},
            },
        })

        if "error" in data:
            return json.dumps(data, ensure_ascii=False)

        results = data.get("results", [])
        formatted = [{
            "title": r.get("title", ""),
            "url": r.get("url", ""),
            "score": round(r.get("score", 0), 3),
            "text": (r.get("text", "") or "")[:1000],
        } for r in results]

        return json.dumps({
            "status": "ok",
            "source_url": url,
            "count": len(formatted),
            "results": formatted,
        }, ensure_ascii=False)
