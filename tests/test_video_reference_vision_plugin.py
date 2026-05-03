from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from astrbot.core.agent.message import TextPart
from astrbot.core.message.components import Reply, Video
from astrbot.core.provider.entities import ProviderRequest


PLUGIN_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"

spec = importlib.util.spec_from_file_location(
    "astrbot_plugin_video_reference_vision_main",
    str(PLUGIN_MAIN_PATH),
)
assert spec and spec.loader
plugin_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = plugin_module
spec.loader.exec_module(plugin_module)


VideoMessageCache = plugin_module.VideoMessageCache
Main = plugin_module.Main
remove_video_text_blocks = plugin_module._remove_video_attachment_text_blocks
extract_video_path = plugin_module._extract_path_from_video_attachment_text


class DummyProvider:
    def __init__(self, provider_config: dict, model: str = "") -> None:
        self.provider_config = provider_config
        self._model = model or str(provider_config.get("model", ""))
        self._key = str(provider_config.get("key", "") or "")

    def get_model(self) -> str:
        return self._model

    def get_current_key(self) -> str:
        return self._key


class DummyContext:
    def __init__(self, provider: DummyProvider) -> None:
        self._provider = provider

    def get_using_provider(self, umo: str | None = None):
        del umo
        return self._provider


def make_event(
    *,
    session_id: str,
    message_id: str,
    message_chain: list,
    sender_id: str = "u1",
    timestamp: int = 123,
):
    msg_obj = SimpleNamespace(
        message_id=message_id,
        message=message_chain,
        sender=SimpleNamespace(user_id=sender_id),
        timestamp=timestamp,
    )
    return SimpleNamespace(unified_msg_origin=session_id, message_obj=msg_obj)


def test_video_cache_put_get_and_expire():
    cache = VideoMessageCache(ttl_seconds=5, max_entries=2)
    cache.put(
        session_id="s1",
        message_id="m1",
        videos=[{"file": "file:///a.mp4", "cover": "", "path": ""}],
        sender_id="u1",
        timestamp=1,
        now_ts=100,
    )
    assert cache.get(session_id="s1", message_id="m1", now_ts=101) is not None
    assert cache.get(session_id="s1", message_id="m1", now_ts=106) is None


def test_remove_video_attachment_text_blocks():
    blocks = [
        {"type": "text", "text": "[Video Attachment: name a.mp4, path /tmp/a.mp4]"},
        {"type": "text", "text": "normal text"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}},
    ]
    cleaned = remove_video_text_blocks(blocks)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "normal text"


@pytest.mark.asyncio
async def test_capture_and_inject_from_reply_chain(tmp_path: Path):
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {
            "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "model": "qwen-vl-max",
        }
    )
    plugin = Main(
        DummyContext(provider),
        config={"enabled": True, "mode": "auto", "max_base64_mb": 10},
    )

    quoted_video = Video.fromFileSystem(str(video_file))
    reply = Reply(id="msg_video_1", chain=[quoted_video])
    llm_event = make_event(
        session_id="platform:group:100",
        message_id="msg_query_2",
        message_chain=[reply],
    )

    req = ProviderRequest(
        prompt="请分析这个视频",
        extra_user_content_parts=[
            TextPart(
                text="[Video Attachment in quoted message: name clip.mp4, path /tmp/clip.mp4]"
            )
        ],
    )

    await plugin.inject_quoted_video(llm_event, req)

    assert req.prompt is None
    assert req.image_urls == []
    assert req.audio_urls == []
    assert req.extra_user_content_parts == []
    assert len(req.contexts) == 1
    content = req.contexts[0]["content"]
    assert any(part.get("type") == "video_url" for part in content)
    assert not any(
        part.get("type") == "text"
        and str(part.get("text", "")).startswith("[Video Attachment")
        for part in content
    )


@pytest.mark.asyncio
async def test_inject_from_reply_id_cache_fallback(tmp_path: Path):
    video_file = tmp_path / "clip2.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {
            "api_base": "https://openrouter.ai/api/v1",
            "model": "openrouter/any-video-model",
        }
    )
    plugin = Main(DummyContext(provider), config={"enabled": True, "mode": "auto"})

    sent_video = Video.fromFileSystem(str(video_file))
    capture_event = make_event(
        session_id="platform:group:200",
        message_id="original_video_msg",
        message_chain=[sent_video],
    )
    await plugin.capture_video_message(capture_event)

    reply_without_chain = Reply(id="original_video_msg", chain=[])
    llm_event = make_event(
        session_id="platform:group:200",
        message_id="query_msg",
        message_chain=[reply_without_chain],
    )
    req = ProviderRequest(prompt="引用视频后提问")

    await plugin.inject_quoted_video(llm_event, req)

    assert len(req.contexts) == 1
    content = req.contexts[0]["content"]
    assert any(part.get("type") == "video_url" for part in content)


@pytest.mark.asyncio
async def test_kimi_upload_strategy_injects_ms_url(tmp_path: Path):
    video_file = tmp_path / "kimi.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    quoted_video = Video.fromFileSystem(str(video_file))
    reply = Reply(id="k1", chain=[quoted_video])

    provider = DummyProvider(
        {
            "api_base": "https://api.moonshot.cn/v1",
            "model": "kimi-k2.5",
            "key": "k_test_key",
        }
    )
    plugin = Main(
        DummyContext(provider),
        config={"enabled": True, "mode": "auto", "kimi_strategy": "upload"},
    )
    event = make_event(
        session_id="platform:group:400",
        message_id="query_2",
        message_chain=[reply],
    )
    req = ProviderRequest(prompt="read this video")

    class FakeFiles:
        async def create(self, file, purpose):
            assert purpose == "video"
            assert str(file).endswith("kimi.mp4")
            return SimpleNamespace(id="file_test_123")

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url):
            assert base_url.startswith("https://api.moonshot.cn")
            assert api_key == "k_test_key"
            self.files = FakeFiles()

    with patch("openai.AsyncOpenAI", FakeAsyncOpenAI):
        await plugin.inject_quoted_video(event, req)

    assert len(req.contexts) == 1
    content = req.contexts[0]["content"]
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("ms://")
        for part in content
    )


def test_extract_path_from_video_attachment_text_windows_style():
    text = (
        "[Video Attachment in quoted message: name demo.mp4, "
        "path D:\\qq data\\clips\\demo test.mp4]"
    )
    path = extract_video_path(text)
    assert path == r"D:\qq data\clips\demo test.mp4"

