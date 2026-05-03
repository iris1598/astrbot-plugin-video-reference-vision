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
extract_video_path = plugin_module._extract_path_from_video_attachment_text


class DummyProvider:
    def __init__(
        self,
        provider_config: dict,
        model: str = "",
        *,
        completion_text: str = "",
    ) -> None:
        self.provider_config = provider_config
        self._model = model or str(provider_config.get("model", ""))
        self._key = str(provider_config.get("key", "") or "")
        self._completion_text = completion_text
        self.calls: list[dict] = []

    def get_model(self) -> str:
        return self._model

    def get_current_key(self) -> str:
        return self._key

    async def text_chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(completion_text=self._completion_text)


class DummyContext:
    def __init__(self, provider: DummyProvider, extra_providers: dict[str, DummyProvider] | None = None) -> None:
        self._provider = provider
        self._extra_providers = extra_providers or {}

    def get_using_provider(self, umo: str | None = None):
        del umo
        return self._provider

    def get_provider_by_id(self, provider_id: str):
        if provider_id == self._provider.provider_config.get("id"):
            return self._provider
        return self._extra_providers.get(provider_id)


class DummyEvent:
    def __init__(
        self,
        *,
        session_id: str,
        message_id: str,
        message_chain: list,
        message_str: str = "",
        sender_id: str = "u1",
        timestamp: int = 123,
    ) -> None:
        self.unified_msg_origin = session_id
        self.message_str = message_str
        self.message_obj = SimpleNamespace(
            message_id=message_id,
            message=message_chain,
            sender=SimpleNamespace(user_id=sender_id),
            timestamp=timestamp,
        )
        self.stopped = False

    def stop_event(self) -> None:
        self.stopped = True


def make_event(**kwargs) -> DummyEvent:
    return DummyEvent(**kwargs)


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


def test_extract_path_from_video_attachment_text_windows_style():
    text = (
        "[Video Attachment in quoted message: name demo.mp4, "
        "path D:\\qq data\\clips\\demo test.mp4]"
    )
    assert extract_video_path(text) == r"D:\qq data\clips\demo test.mp4"


@pytest.mark.asyncio
async def test_mode_off_does_not_intercept_direct_video(tmp_path: Path):
    video_file = tmp_path / "off.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "off",
            "allow_direct_video": False,
            "intercept_direct_video_llm_request": True,
        },
    )

    event = make_event(
        session_id="platform:group:100",
        message_id="msg_off",
        message_chain=[Video.fromFileSystem(str(video_file))],
        message_str="帮我看看这是什么",
    )
    req = ProviderRequest(prompt="帮我看看这是什么")

    await plugin.inject_quoted_video(event, req)

    assert event.stopped is False
    assert req.contexts == []
    assert req.prompt == "帮我看看这是什么"


@pytest.mark.asyncio
async def test_provider_denylist_prevents_direct_intercept(tmp_path: Path):
    video_file = tmp_path / "deny.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "provider_denylist": ["qwen-vl-max"],
        },
    )

    event = make_event(
        session_id="platform:group:101",
        message_id="msg_deny",
        message_chain=[Video.fromFileSystem(str(video_file))],
        message_str="帮我看看这是什么",
    )
    req = ProviderRequest(prompt="帮我看看这是什么")

    await plugin.inject_quoted_video(event, req)

    assert event.stopped is False
    assert req.contexts == []


@pytest.mark.asyncio
async def test_direct_video_only_request_is_intercepted(tmp_path: Path):
    video_file = tmp_path / "direct.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "allow_direct_video": False,
            "intercept_direct_video_llm_request": True,
        },
    )

    event = make_event(
        session_id="platform:group:102",
        message_id="msg_direct",
        message_chain=[Video.fromFileSystem(str(video_file))],
        message_str="",
    )
    req = ProviderRequest(
        extra_user_content_parts=[
            TextPart(text=f"[Video Attachment: name direct.mp4, path {video_file}]")
        ]
    )

    await plugin.inject_quoted_video(event, req)

    assert event.stopped is True
    assert req.contexts == []


@pytest.mark.asyncio
async def test_direct_video_with_text_is_also_intercepted(tmp_path: Path):
    video_file = tmp_path / "direct_text.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "allow_direct_video": False,
            "intercept_direct_video_llm_request": True,
        },
    )

    event = make_event(
        session_id="platform:group:103",
        message_id="msg_direct_text",
        message_chain=[Video.fromFileSystem(str(video_file))],
        message_str="帮我看看这是什么",
    )
    req = ProviderRequest(prompt="帮我看看这是什么")

    await plugin.inject_quoted_video(event, req)

    assert event.stopped is True
    assert req.contexts == []
    assert req.prompt == "帮我看看这是什么"


@pytest.mark.asyncio
async def test_capture_and_inject_from_reply_chain_rewrites_request(tmp_path: Path):
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={"enabled": True, "mode": "auto", "max_base64_mb": 10},
    )

    reply = Reply(id="msg_video_1", chain=[Video.fromFileSystem(str(video_file))])
    event = make_event(
        session_id="platform:group:104",
        message_id="msg_query_2",
        message_chain=[reply],
        message_str="请分析这个视频",
    )
    req = ProviderRequest(
        prompt="请分析这个视频",
        extra_user_content_parts=[
            TextPart(
                text="[Video Attachment in quoted message: name clip.mp4, path /tmp/clip.mp4]"
            )
        ],
    )

    await plugin.inject_quoted_video(event, req)

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
        {"id": "chat_openrouter", "api_base": "https://openrouter.ai/api/v1", "model": "openrouter/any-video-model"}
    )
    plugin = Main(DummyContext(provider), config={"enabled": True, "mode": "auto"})

    capture_event = make_event(
        session_id="platform:group:105",
        message_id="original_video_msg",
        message_chain=[Video.fromFileSystem(str(video_file))],
    )
    await plugin.capture_video_message(capture_event)

    event = make_event(
        session_id="platform:group:105",
        message_id="query_msg",
        message_chain=[Reply(id="original_video_msg", chain=[])],
        message_str="引用视频后提问",
    )
    req = ProviderRequest(prompt="引用视频后提问")

    await plugin.inject_quoted_video(event, req)

    assert len(req.contexts) == 1
    assert any(part.get("type") == "video_url" for part in req.contexts[0]["content"])


@pytest.mark.asyncio
async def test_provider_allowlist_blocks_non_matching_provider(tmp_path: Path):
    video_file = tmp_path / "clip3.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "provider_allowlist": ["kimi"],
        },
    )

    event = make_event(
        session_id="platform:group:106",
        message_id="query_1",
        message_chain=[Reply(id="r1", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="test",
    )
    req = ProviderRequest(prompt="test")

    await plugin.inject_quoted_video(event, req)

    assert req.contexts == []
    assert req.prompt == "test"


@pytest.mark.asyncio
async def test_kimi_auto_uses_base64_for_small_local_video(tmp_path: Path):
    video_file = tmp_path / "kimi_small.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_kimi", "api_base": "https://api.moonshot.cn/v1", "model": "kimi-k2.5", "key": "k_test_key"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "kimi_strategy": "auto",
            "max_base64_mb": 10,
        },
    )

    event = make_event(
        session_id="platform:group:107",
        message_id="query_2",
        message_chain=[Reply(id="k1", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="read this video",
    )
    req = ProviderRequest(prompt="read this video")

    class FailAsyncOpenAI:
        def __init__(self, *args, **kwargs):
            raise AssertionError("small local Kimi video should not upload in auto mode")

    with patch("openai.AsyncOpenAI", FailAsyncOpenAI):
        await plugin.inject_quoted_video(event, req)

    assert len(req.contexts) == 1
    content = req.contexts[0]["content"]
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("data:video/")
        for part in content
    )


@pytest.mark.asyncio
async def test_kimi_auto_uploads_oversized_local_video(tmp_path: Path):
    video_file = tmp_path / "kimi_big.mp4"
    video_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

    provider = DummyProvider(
        {"id": "chat_kimi", "api_base": "https://api.moonshot.cn/v1", "model": "kimi-k2.5", "key": "k_test_key"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "kimi_strategy": "auto",
            "max_base64_mb": 1,
            "kimi_upload_on_oversize": True,
        },
    )

    event = make_event(
        session_id="platform:group:108",
        message_id="query_3",
        message_chain=[Reply(id="k2", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="read this video",
    )
    req = ProviderRequest(prompt="read this video")

    class FakeFiles:
        async def create(self, file, purpose):
            assert purpose == "video"
            return SimpleNamespace(id="file_test_oversize")

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url):
            assert api_key == "k_test_key"
            assert base_url.startswith("https://api.moonshot.cn")
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


@pytest.mark.asyncio
async def test_kimi_explicit_upload_overrides_public_url(tmp_path: Path):
    local_video = tmp_path / "downloaded.mp4"
    local_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_kimi", "api_base": "https://api.moonshot.cn/v1", "model": "kimi-k2.5", "key": "k_test_key"}
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "kimi_strategy": "upload",
            "prefer_public_url": True,
        },
    )

    remote_video = Video(file="https://example.com/demo.mp4")

    async def fake_convert_to_file_path(self):
        del self
        return str(local_video)

    event = make_event(
        session_id="platform:group:109",
        message_id="query_4",
        message_chain=[Reply(id="k3", chain=[remote_video])],
        message_str="read this remote video",
    )
    req = ProviderRequest(prompt="read this remote video")

    class FakeFiles:
        async def create(self, file, purpose):
            assert Path(file) == local_video
            assert purpose == "video"
            return SimpleNamespace(id="file_test_remote_upload")

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url):
            self.files = FakeFiles()

    with patch.object(Video, "convert_to_file_path", fake_convert_to_file_path), patch(
        "openai.AsyncOpenAI", FakeAsyncOpenAI
    ):
        await plugin.inject_quoted_video(event, req)

    assert len(req.contexts) == 1
    content = req.contexts[0]["content"]
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("ms://")
        for part in content
    )


@pytest.mark.asyncio
async def test_video_caption_provider_rewrites_request_as_text_summary(tmp_path: Path):
    video_file = tmp_path / "caption.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    chat_provider = DummyProvider(
        {"id": "chat_text", "api_base": "https://api.example.com/v1", "provider": "openai_chat_completion", "model": "text-only-model"}
    )
    caption_provider = DummyProvider(
        {"id": "video_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"},
        completion_text="视频里有人在演示插件配置页面。",
    )
    plugin = Main(
        DummyContext(chat_provider, {"video_qwen": caption_provider}),
        config={
            "enabled": True,
            "mode": "auto",
            "video_caption_provider_id": "video_qwen",
            "video_caption_prompt": "请帮我转述这个视频",
            "video_caption_use_current_question": True,
        },
    )

    event = make_event(
        session_id="platform:group:110",
        message_id="query_5",
        message_chain=[Reply(id="r2", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="这个视频在讲什么？",
    )
    req = ProviderRequest(prompt="这个视频在讲什么？")

    await plugin.inject_quoted_video(event, req)

    assert len(caption_provider.calls) == 1
    caption_contexts = caption_provider.calls[0]["contexts"]
    user_content = caption_contexts[0]["content"]
    assert any(part.get("type") == "video_url" for part in user_content)
    assert any(
        part.get("type") == "text" and "用户当前问题：这个视频在讲什么？" in part.get("text", "")
        for part in user_content
    )
    assert len(req.contexts) == 1
    rewritten = req.contexts[0]["content"]
    assert any(
        part.get("type") == "text"
        and "[引用视频内容转述]" in part.get("text", "")
        and "视频里有人在演示插件配置页面。" in part.get("text", "")
        for part in rewritten
    )
    assert not any(part.get("type") == "video_url" for part in rewritten)


def test_global_llm_metadata_is_used_for_strategy_detection():
    provider = DummyProvider(
        {
            "id": "chat_custom",
            "provider": "openai_chat_completion",
            "api_base": "https://example.com/v1",
            "modalities": None,
        },
        model="custom-video-model",
    )

    with patch.dict(
        plugin_module.LLM_METADATAS,
        {
            "custom-video-model": {
                "id": "custom-video-model",
                "reasoning": False,
                "tool_call": False,
                "knowledge": "none",
                "release_date": "",
                "modalities": {"input": ["text", "video"], "output": ["text"]},
                "open_weights": False,
                "limit": {"context": 0, "output": 0},
            }
        },
        clear=False,
    ):
        strategy = plugin_module._detect_video_strategy(
            provider,
            mode="auto",
            prefer_model_metadata_video=True,
        )

    assert strategy == "generic"
