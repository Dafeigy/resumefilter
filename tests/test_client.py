import json

import httpx
import pytest

from resumefilter.client import OCRError, SiliconFlowOCRClient
from resumefilter.config import OCRSettings


def settings(**overrides: object) -> OCRSettings:
    values = {
        "api_key": "secret",
        "api_url": "https://api.siliconflow.cn/v1/chat/completions",
        "model": "deepseek-ai/DeepSeek-OCR",
        "max_retries": 1,
    }
    values.update(overrides)
    return OCRSettings(**values)


@pytest.mark.asyncio
async def test_request_payload_retry_and_output_cleaning() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(429, headers={"Retry-After": "0"})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "<|ref|>title<|/ref|><|det|>[[1,2,3,4]]<|/det|>\n# 张三\n<｜end▁of▁sentence｜>"
                        },
                    }
                ]
            },
        )

    client = SiliconFlowOCRClient(settings(), transport=httpx.MockTransport(handler))
    try:
        result = await client.recognize(b"image", "image/png", "request-id")
    finally:
        await client.close()

    assert result == "# 张三"
    assert len(requests) == 2
    payload = json.loads(requests[-1].content)
    assert payload["model"] == "deepseek-ai/DeepSeek-OCR"
    assert payload["messages"][0]["content"][0]["image_url"]["url"].startswith(
        "data:image/png;base64,"
    )
    assert requests[-1].headers["authorization"] == "Bearer secret"


@pytest.mark.asyncio
async def test_length_truncation_is_an_error() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"finish_reason": "length", "message": {"content": "partial"}}
                ]
            },
        )

    client = SiliconFlowOCRClient(
        settings(max_retries=0), transport=httpx.MockTransport(handler)
    )
    try:
        with pytest.raises(OCRError, match="context limit"):
            await client.recognize(b"image", "image/png")
    finally:
        await client.close()

