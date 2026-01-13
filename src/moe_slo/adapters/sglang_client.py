# src/moe_slo/adapters/sglang_client.py
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Optional

import httpx


def now_s() -> float:
    return time.time()


@dataclass
class RequestRecord:
    req_id: str
    send_ts: float
    first_ts: float
    last_ts: float
    # 可选：未来你可以把 prompt/output token 数补上
    prompt_tokens: Optional[int] = None
    output_tokens: Optional[int] = None

    @property
    def ttft_s(self) -> float:
        return self.first_ts - self.send_ts

    @property
    def e2e_s(self) -> float:
        return self.last_ts - self.send_ts


class SGLangClient:
    """
    Minimal streaming client to an OpenAI-compatible Chat Completions endpoint.

    Default assumes SSE format:
      data: {...json...}
      data: {...json...}
      data: [DONE]
    """

    def __init__(
        self,
        base_url: str,
        endpoint: str = "/v1/chat/completions",
        timeout: Optional[float] = None,
    ) -> None:
        self.url = base_url.rstrip("/") + endpoint
        self.timeout = timeout

    async def _iter_sse_lines(self, r: httpx.Response) -> AsyncIterator[str]:
        """
        Yields raw lines from the HTTP response stream.
        """
        async for line in r.aiter_lines():
            if line:
                yield line

    def _is_done_sse(self, data_str: str) -> bool:
        return data_str.strip() == "[DONE]"

    def _parse_sse_payload(self, line: str) -> Optional[str]:
        """
        For SSE, lines usually look like:
          data: {...}
        Return the part after "data: " or None if not a data line.
        """
        if line.startswith("data:"):
            return line[len("data:"):].strip()
        return None

    async def generate_stream(
        self,
        req_id: str,
        payload: Dict[str, Any],
        client: httpx.AsyncClient,
    ) -> RequestRecord:
        """
        Sends a single streaming request and returns RequestRecord with send/first/last timestamps.
        """
        send_ts = now_s()
        first_ts: Optional[float] = None
        last_ts: Optional[float] = None

        # 强制 stream=True（如果你 payload 已含可忽略）
        payload = dict(payload)
        payload["stream"] = True

        async with client.stream("POST", self.url, json=payload, timeout=self.timeout) as r:
            r.raise_for_status()

            async for line in self._iter_sse_lines(r):
                data = self._parse_sse_payload(line)
                if data is None:
                    # 某些实现会混入 event: / id: 等行，忽略
                    continue

                if self._is_done_sse(data):
                    last_ts = now_s()
                    break

                # 第一次收到任何 token chunk（或 delta chunk）就记 TTFT
                if first_ts is None:
                    first_ts = now_s()

                # 如果你想从 data JSON 中解析 output_tokens，可在这里做
                # obj = json.loads(data)

        if last_ts is None:
            last_ts = now_s()
        if first_ts is None:
            first_ts = last_ts  # 没流式数据也不崩

        return RequestRecord(req_id=req_id, send_ts=send_ts, first_ts=first_ts, last_ts=last_ts)
