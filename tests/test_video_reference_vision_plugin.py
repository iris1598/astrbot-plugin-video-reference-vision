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
detect_video_strategy = plugin_module._detect_video_strategy


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


async def _assembled_content(req: ProviderRequest) -> list[dict]:
    return (await req.assemble_context())["content"]


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

    assert req.prompt == "请分析这个视频"
    assert req.image_urls == []
    assert req.audio_urls == []
    assert req.contexts == []
    content = await _assembled_content(req)
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

    assert req.contexts == []
    assert any(part.get("type") == "video_url" for part in await _assembled_content(req))


@pytest.mark.asyncio
async def test_reply_chain_video_with_invalid_file_uses_path_fallback(tmp_path: Path):
    video_file = tmp_path / "qq_cached.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(DummyContext(provider), config={"enabled": True, "mode": "auto"})

    reply_video = Video(file="894034488f5679dc30046b8e1af3746a.mp4", path=str(video_file))
    event = make_event(
        session_id="platform:group:105b",
        message_id="query_msg_path_fallback",
        message_chain=[Reply(id="quoted_qq_video", chain=[reply_video])],
        message_str="引用视频后提问",
    )
    req = ProviderRequest(prompt="引用视频后提问")

    await plugin.inject_quoted_video(event, req)

    assert req.contexts == []
    assert any(part.get("type") == "video_url" for part in await _assembled_content(req))


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
async def test_invalid_video_file_without_fallback_does_not_raise(tmp_path: Path):
    provider = DummyProvider(
        {"id": "chat_qwen", "api_base": "https://dashscope.aliyuncs.com/compatible-mode/v1", "model": "qwen-vl-max"}
    )
    plugin = Main(DummyContext(provider), config={"enabled": True, "mode": "auto"})

    reply_video = Video(file="894034488f5679dc30046b8e1af3746a.mp4", path="")
    event = make_event(
        session_id="platform:group:106b",
        message_id="query_invalid_video",
        message_chain=[Reply(id="quoted_invalid_video", chain=[reply_video])],
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

    assert req.contexts == []
    content = await _assembled_content(req)
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

    assert req.contexts == []
    content = await _assembled_content(req)
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

    assert req.contexts == []
    content = await _assembled_content(req)
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("ms://")
        for part in content
    )


@pytest.mark.asyncio
async def test_opencode_kimi_remote_video_uses_base64_not_public_url(tmp_path: Path):
    local_video = tmp_path / "opencode_kimi.mp4"
    local_video.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {
            "id": "chat_opencode_kimi",
            "api_base": "https://opencode.ai/zen/go/v1/chat/completions",
            "model": "opencode-go/kimi-k2.6",
        }
    )
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "prefer_public_url": True,
            "kimi_strategy": "auto",
            "max_base64_mb": 10,
        },
    )

    remote_video = Video(file="https://example.com/opencode-kimi.mp4")

    async def fake_convert_to_file_path(self):
        del self
        return str(local_video)

    event = make_event(
        session_id="platform:group:109b",
        message_id="query_4b",
        message_chain=[Reply(id="k3b", chain=[remote_video])],
        message_str="read this remote video",
    )
    req = ProviderRequest(prompt="read this remote video")

    with patch.object(Video, "convert_to_file_path", fake_convert_to_file_path):
        await plugin.inject_quoted_video(event, req)

    assert req.contexts == []
    content = await _assembled_content(req)
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("data:video/")
        for part in content
    )
    assert not any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("https://")
        for part in content
    )


@pytest.mark.asyncio
async def test_kimicode_base_url_uploads_oversized_local_video(tmp_path: Path):
    video_file = tmp_path / "kimi_code_big.mp4"
    video_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

    provider = DummyProvider(
        {
            "id": "chat_kimi_code",
            "api_base": "https://api.kimi.com/coding/v1",
            "model": "kimi-k2.6",
            "key": "kc_test_key",
        }
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
        session_id="platform:group:109c",
        message_id="query_4c",
        message_chain=[Reply(id="k3c", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="read this video",
    )
    req = ProviderRequest(prompt="read this video")

    class FakeFiles:
        async def create(self, file, purpose):
            assert purpose == "video"
            return SimpleNamespace(id="file_test_kimicode")

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url):
            assert api_key == "kc_test_key"
            assert base_url == "https://api.kimi.com/coding/v1"
            self.files = FakeFiles()

    with patch("openai.AsyncOpenAI", FakeAsyncOpenAI):
        await plugin.inject_quoted_video(event, req)

    content = await _assembled_content(req)
    assert any(
        part.get("type") == "video_url"
        and str(part.get("video_url", {}).get("url", "")).startswith("ms://")
        for part in content
    )


@pytest.mark.asyncio
async def test_direct_transport_kimicode_forces_kimi_upload_on_custom_base(tmp_path: Path):
    video_file = tmp_path / "direct_kimicode_big.mp4"
    video_file.write_bytes(b"\x00" * (2 * 1024 * 1024))

    chat_provider = DummyProvider(
        {"id": "chat_text", "api_base": "https://api.example.com/v1", "model": "text-only-model"}
    )
    plugin = Main(
        DummyContext(chat_provider),
        config={
            "enabled": True,
            "mode": "auto",
            "kimi_strategy": "auto",
            "max_base64_mb": 1,
            "kimi_upload_on_oversize": True,
            "video_caption_direct_enabled": True,
            "video_caption_direct_transport": "kimicode",
            "video_caption_direct_base_url": "https://proxy.example.com/v1",
            "video_caption_direct_api_key": "direct_test_key",
            "video_caption_direct_model": "kimi-k2.6",
        },
    )

    provider = plugin._build_direct_caption_provider()
    assert provider is not None
    assert detect_video_strategy(provider, mode="force") == "kimi"

    class FakeFiles:
        async def create(self, file, purpose):
            assert purpose == "video"
            return SimpleNamespace(id="file_test_direct_kimicode")

    class FakeAsyncOpenAI:
        def __init__(self, api_key, base_url):
            assert api_key == "direct_test_key"
            assert base_url == "https://proxy.example.com/v1"
            self.files = FakeFiles()

    with patch("openai.AsyncOpenAI", FakeAsyncOpenAI):
        part = await plugin._build_video_part(
            Video.fromFileSystem(str(video_file)),
            strategy=detect_video_strategy(provider, mode="force"),
            provider=provider,
        )

    assert part is not None
    assert str(part.get("video_url", {}).get("url", "")).startswith("ms://")


def test_direct_transport_generic_disables_kimi_transport_override():
    provider = plugin_module.DirectCaptionProvider(
        provider_id="__video_caption_direct__",
        transport="generic",
        api_base="https://proxy.example.com/v1",
        api_key="direct_test_key",
        model="kimi-k2.6",
        timeout_seconds=30,
    )

    assert detect_video_strategy(provider, mode="force") == "generic"


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

    with patch.object(plugin_module.logger, "info") as mock_logger_info:
        await plugin.inject_quoted_video(event, req)

    assert len(caption_provider.calls) == 1
    caption_contexts = caption_provider.calls[0]["contexts"]
    user_content = caption_contexts[0]["content"]
    assert any(part.get("type") == "video_url" for part in user_content)
    assert any(
        part.get("type") == "text" and "用户当前问题：这个视频在讲什么？" in part.get("text", "")
        for part in user_content
    )
    assert req.contexts == []
    assert req.prompt == event.message_str
    rewritten = await _assembled_content(req)
    assert any(
        part.get("type") == "text"
        and "[引用视频内容转述]" in part.get("text", "")
        and "视频里有人在演示插件配置页面。" in part.get("text", "")
        for part in rewritten
    )
    assert not any(part.get("type") == "video_url" for part in rewritten)
    mock_logger_info.assert_any_call(
        "video-reference-vision: caption summary: %s",
        "视频里有人在演示插件配置页面。",
    )


@pytest.mark.asyncio
async def test_current_provider_falls_back_to_frame_caption_when_video_is_rejected(tmp_path: Path):
    video_file = tmp_path / "frames.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_kimi", "api_base": "https://api.moonshot.cn/v1", "model": "kimi-k2.5"}
    )

    async def fake_text_chat(**kwargs):
        provider.calls.append(kwargs)
        content = kwargs["contexts"][0]["content"]
        if any(part.get("type") == "video_url" for part in content):
            raise RuntimeError("No endpoints found that support input video")
        assert any(part.get("type") == "image_url" for part in content)
        return SimpleNamespace(completion_text="frame based summary")

    provider.text_chat = fake_text_chat
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "max_base64_mb": 10,
        },
    )

    event = make_event(
        session_id="platform:group:111",
        message_id="query_frame_fallback",
        message_chain=[Reply(id="r_frame", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="这个视频在讲什么？",
    )
    req = ProviderRequest(prompt="这个视频在讲什么？")

    with patch.object(
        plugin,
        "_extract_frame_data_urls",
        return_value=["data:image/jpeg;base64,AAAA"],
    ):
        await plugin.inject_quoted_video(event, req)

    assert len(provider.calls) == 2
    assert req.contexts == []
    assert req.prompt == event.message_str
    rewritten = await _assembled_content(req)
    assert any(
        part.get("type") == "text"
        and "frame based summary" in part.get("text", "")
        for part in rewritten
    )
    assert not any(part.get("type") == "video_url" for part in rewritten)


@pytest.mark.asyncio
async def test_current_provider_media_rejection_skips_native_video_injection(tmp_path: Path):
    video_file = tmp_path / "reject_all.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "chat_kimi", "api_base": "https://api.moonshot.cn/v1", "model": "kimi-k2.5"}
    )

    async def fake_text_chat(**kwargs):
        provider.calls.append(kwargs)
        content = kwargs["contexts"][0]["content"]
        if any(part.get("type") == "video_url" for part in content):
            raise RuntimeError("No endpoints found that support input video")
        raise RuntimeError("supported types: ['text']")

    provider.text_chat = fake_text_chat
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "max_base64_mb": 10,
        },
    )

    event = make_event(
        session_id="platform:group:112",
        message_id="query_skip_injection",
        message_chain=[Reply(id="r_skip", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="这个视频在讲什么？",
    )
    req = ProviderRequest(prompt="这个视频在讲什么？")

    with patch.object(
        plugin,
        "_extract_frame_data_urls",
        return_value=["data:image/jpeg;base64,BBBB"],
    ):
        await plugin.inject_quoted_video(event, req)

    assert len(provider.calls) == 2
    assert req.contexts == []
    assert req.prompt == "这个视频在讲什么？"


@pytest.mark.asyncio
async def test_explicit_current_caption_provider_id_still_skips_native_video_injection(tmp_path: Path):
    video_file = tmp_path / "reject_same_provider.mp4"
    video_file.write_bytes(b"\x00\x00\x00\x18ftypmp42")

    provider = DummyProvider(
        {"id": "QINGYI-Normal", "api_base": "https://api.example.com/v1", "model": "kimi-k2.5"}
    )

    async def fake_text_chat(**kwargs):
        provider.calls.append(kwargs)
        content = kwargs["contexts"][0]["content"]
        if any(part.get("type") == "video_url" for part in content):
            raise RuntimeError("Error code: 400 - {'error': {'message': 'Error from provider: No endpoints found that support input video', 'code': 404}}")
        raise RuntimeError("ffmpeg fallback unavailable")

    provider.text_chat = fake_text_chat
    plugin = Main(
        DummyContext(provider),
        config={
            "enabled": True,
            "mode": "auto",
            "video_caption_provider_id": "QINGYI-Normal",
            "max_base64_mb": 10,
        },
    )

    event = make_event(
        session_id="platform:group:113",
        message_id="query_same_provider_skip",
        message_chain=[Reply(id="r_same", chain=[Video.fromFileSystem(str(video_file))])],
        message_str="这个视频在讲什么？",
    )
    req = ProviderRequest(prompt="这个视频在讲什么？")

    with patch.object(plugin, "_extract_frame_data_urls", return_value=[]):
        await plugin.inject_quoted_video(event, req)

    assert len(provider.calls) == 1
    assert req.contexts == []
    assert req.prompt == "这个视频在讲什么？"


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
