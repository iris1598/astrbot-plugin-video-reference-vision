from __future__ import annotations

import asyncio
import base64
import glob
import json
import math
import mimetypes
import os
import platform
import re
import shutil
import socket
import subprocess
import tempfile
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import urlparse
import uuid

from openai import BaseModel as OpenAIBaseModel
from pydantic import BaseModel

from astrbot import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.message_components import Image, Reply, Video
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star
from astrbot.core.agent.message import ContentPart, ImageURLPart, TextPart
from astrbot.core.utils.llm_metadata import LLM_METADATAS


DEFAULT_VIDEO_CAPTION_PROMPT = (
    "请阅读这个视频或 GIF，并用中文转述与用户问题直接相关的内容。"
    "如果用户没有提出具体问题，就概括主要内容、关键画面、对白或字幕，以及事件顺序。"
    "不要编造没有出现的信息。"
)

VIDEO_TRANSPORT_AUTO = "auto"
VIDEO_TRANSPORT_GENERIC = "generic"
VIDEO_TRANSPORT_KIMI_MOONSHOT = "moonshot"
VIDEO_TRANSPORT_KIMI_KIMICODE = "kimicode"
KIMI_CODE_MODEL_ID = "kimi-for-coding"
KIMI_VIDEO_TRANSPORTS = {
    VIDEO_TRANSPORT_KIMI_MOONSHOT,
    VIDEO_TRANSPORT_KIMI_KIMICODE,
}
FRAME_MODE_AUTO = "auto"
FRAME_MODE_COUNT = "count"
FRAME_MODE_FPS = "fps"
AUTO_FRAME_DEFAULT_TOKENS = 1200
AUTO_FRAME_BASE64_CHARS_PER_TOKEN = 2.0
AUTO_FRAME_TEXT_TOKENS = 3000
AUTO_FRAME_MIN_FRAME_TOKENS = 800


DEFAULT_CONFIG: dict[str, Any] = {
    "enabled": True,
    "mode": "auto",  # auto | force | off
    "enable_onebot_media_resolver": True,
    "prefer_public_url": True,
    "max_base64_mb": 20,
    "fallback_behavior": "keep_text",  # keep_text | silent
    "prefer_model_metadata_video": True,
    "qwen_fps": 2.0,
    "generic_fps": 2.0,
    "kimi_strategy": "auto",  # auto | upload | base64
    "kimi_upload_on_oversize": True,
    "kimi_api_base": "",
    "max_videos_per_message": 3,
    "max_videos_per_request": 1,
    "cache_ttl_seconds": 7200,
    "cache_max_entries": 500,
    "remove_default_video_text": True,
    "allow_direct_video": False,
    "intercept_direct_video_llm_request": True,
    "provider_allowlist": [],
    "provider_denylist": [],
    "video_caption_provider_id": "",
    "video_caption_prompt": DEFAULT_VIDEO_CAPTION_PROMPT,
    "video_caption_use_current_question": True,
    "video_caption_use_current_provider": True,
    "video_caption_frame_fallback": True,
    "video_caption_frame_mode": "auto",
    "video_caption_frame_count": 4,
    "video_caption_frame_fps": 1.0,
    "video_caption_frame_auto_min_context_k": 150,
    "video_caption_frame_auto_max_context_k": 200,
    "video_caption_context_cache_rounds": 0,
    "native_video_injection_fallback": True,
    "video_caption_direct_enabled": False,
    "video_caption_direct_transport": VIDEO_TRANSPORT_AUTO,
    "video_caption_direct_base_url": "",
    "video_caption_direct_api_key": "",
    "video_caption_direct_model": "",
    "video_caption_direct_timeout_seconds": 120,
    "ffmpeg_path": "",
    "ffprobe_path": "",
    "enable_gif_input": False,
}


SupportedMedia = Video | Image


class VideoURLPart(ContentPart):
    class VideoURL(BaseModel):
        url: str
        id: str | None = None

    type: str = "video_url"
    video_url: VideoURL
    fps: float | None = None


class KimiFileObject(OpenAIBaseModel):
    id: str


def _normalized_message_id(value: Any) -> str:
    return str(value or "").strip()


def _is_http_url(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _provider_api_base(provider: Any) -> str:
    provider_config = getattr(provider, "provider_config", {}) or {}
    return str(provider_config.get("api_base", "") or "").strip().lower()


def _normalize_openai_base_url(value: Any) -> str:
    base_url = str(value or "").strip()
    if not base_url:
        return ""
    for suffix in (
        "/chat/completions",
        "/responses",
        "/completions",
        "/messages",
    ):
        if base_url.endswith(suffix):
            return base_url[: -len(suffix)].rstrip("/")
    return base_url.rstrip("/")


def _normalize_video_transport(value: Any) -> str:
    transport = str(value or "").strip().lower()
    if transport in {
        VIDEO_TRANSPORT_AUTO,
        VIDEO_TRANSPORT_GENERIC,
        VIDEO_TRANSPORT_KIMI_MOONSHOT,
        VIDEO_TRANSPORT_KIMI_KIMICODE,
    }:
        return transport
    return VIDEO_TRANSPORT_AUTO


def _is_kimicode_api_base(value: Any) -> bool:
    return "api.kimi.com/coding" in str(value or "").strip().lower()


def _uses_kimicode_transport(*, transport: str, api_base: Any) -> bool:
    normalized_transport = _normalize_video_transport(transport)
    if normalized_transport == VIDEO_TRANSPORT_KIMI_KIMICODE:
        return True
    if normalized_transport == VIDEO_TRANSPORT_AUTO and _is_kimicode_api_base(api_base):
        return True
    return False


def _normalize_direct_caption_model(
    *,
    transport: str,
    api_base: Any,
    model: Any,
) -> str:
    if _uses_kimicode_transport(transport=transport, api_base=api_base):
        return KIMI_CODE_MODEL_ID
    return str(model or "").strip()


def _ascii_header_value(value: Any, *, fallback: str = "unknown") -> str:
    text = str(value or "").strip()
    if not text:
        return fallback
    try:
        text.encode("ascii")
        return text
    except UnicodeEncodeError:
        normalized = text.encode("ascii", errors="ignore").decode("ascii").strip()
        return normalized or fallback


def _kimi_compat_headers(provider: Any) -> dict[str, str]:
    transport = _provider_video_transport(provider)
    if transport != VIDEO_TRANSPORT_KIMI_KIMICODE:
        api_base = _provider_api_base(provider)
        if "api.kimi.com/coding" not in api_base:
            return {}
    return {
        "User-Agent": "KimiCLI/compat",
        "X-Msh-Platform": "kimi_cli",
        "X-Msh-Version": "compat",
        "X-Msh-Device-Name": _ascii_header_value(socket.gethostname()),
        "X-Msh-Device-Model": _ascii_header_value(platform.machine()),
        "X-Msh-Os-Version": _ascii_header_value(platform.version()),
        "X-Msh-Device-Id": _ascii_header_value(uuid.getnode(), fallback="0"),
    }


def _provider_video_transport(provider: Any) -> str:
    provider_config = getattr(provider, "provider_config", {}) or {}
    return _normalize_video_transport(provider_config.get("video_transport"))


def _is_kimicode_provider(provider: Any, *, strategy: str | None = None) -> bool:
    if strategy != "kimi":
        return False
    return _uses_kimicode_transport(
        transport=_provider_video_transport(provider),
        api_base=_provider_api_base(provider),
    )


def _supports_kimi_file_upload(provider: Any) -> bool:
    transport = _provider_video_transport(provider)
    if transport in KIMI_VIDEO_TRANSPORTS:
        return True
    api_base = _provider_api_base(provider)
    return (
        "moonshot" in api_base
        or "kimi.ai" in api_base
        or "api.kimi.com/coding" in api_base
    )


def _is_video_attachment_text(text: str) -> bool:
    return text.startswith("[Video Attachment")


_VIDEO_ATTACHMENT_PATH_RE = re.compile(
    r"^\[Video Attachment(?: in quoted message)?: name .*?, path (?P<path>.+)\]$"
)


def _extract_path_from_video_attachment_text(text: str) -> str | None:
    if not _is_video_attachment_text(text):
        return None
    match = _VIDEO_ATTACHMENT_PATH_RE.match(text.strip())
    if not match:
        return None
    path = str(match.group("path") or "").strip()
    return path or None


def _coerce_content_blocks(content: Any) -> list[dict]:
    if isinstance(content, list):
        return list(content)
    if isinstance(content, str):
        text = content.strip()
        if text:
            return [{"type": "text", "text": text}]
    return []


def _get_provider_modalities(provider: Any) -> list[str]:
    provider_config = getattr(provider, "provider_config", {}) or {}
    modalities = provider_config.get("modalities")
    if isinstance(modalities, list):
        return [str(x).lower() for x in modalities]

    model_metadata = provider_config.get("model_metadata")
    if isinstance(model_metadata, dict):
        inputs = model_metadata.get("modalities", {}).get("input", [])
        if isinstance(inputs, list):
            return [str(x).lower() for x in inputs]

    model_name = str(getattr(provider, "get_model", lambda: "")() or "").strip()
    if model_name:
        metadata = LLM_METADATAS.get(model_name)
        if isinstance(metadata, dict):
            inputs = metadata.get("modalities", {}).get("input", [])
            if isinstance(inputs, list):
                return [str(x).lower() for x in inputs]
    return []


def _remove_video_attachment_text_blocks(content_blocks: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for block in content_blocks:
        if not isinstance(block, dict):
            cleaned.append(block)
            continue
        if block.get("type") != "text":
            cleaned.append(block)
            continue
        if _is_video_attachment_text(str(block.get("text", ""))):
            continue
        cleaned.append(block)
    return cleaned


def _remove_video_attachment_text_from_extra_parts(extra_parts: list[Any]) -> list[Any]:
    cleaned: list[Any] = []
    for part in extra_parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and _is_video_attachment_text(text):
            continue
        cleaned.append(part)
    return cleaned


def _detect_video_strategy(
    provider: Any,
    mode: str,
    *,
    prefer_model_metadata_video: bool = True,
) -> str | None:
    if mode == "off":
        return None

    provider_config = getattr(provider, "provider_config", {}) or {}
    model = str(getattr(provider, "get_model", lambda: "")() or "").lower()
    api_base = _provider_api_base(provider)
    provider_name = str(provider_config.get("provider", "") or "").lower()
    transport = _provider_video_transport(provider)

    if transport in KIMI_VIDEO_TRANSPORTS:
        return "kimi"
    if transport == VIDEO_TRANSPORT_GENERIC:
        return "generic"

    if any(token in api_base for token in ("moonshot", "kimi")) or "kimi" in model:
        return "kimi"
    if "dashscope" in api_base or "qwen" in model or "qvq" in model:
        return "qwen"
    if "openrouter.ai" in api_base:
        return "openrouter"
    if "mimo" in model or "xiaomi" in model:
        return "openrouter"

    modalities = _get_provider_modalities(provider) if prefer_model_metadata_video else []
    if "video" in modalities:
        return "generic"

    if provider_name in {"openai", "openai_chat_completion"}:
        return "generic" if mode == "force" else None
    return "generic" if mode == "force" else None


def _guess_mime(path: str) -> str:
    mime_type = mimetypes.guess_type(path)[0]
    if mime_type and (mime_type.startswith("video/") or mime_type == "image/gif"):
        return mime_type
    return "video/mp4"


def _guess_kimi_upload_filename(mime_type: str) -> str:
    extension = mimetypes.guess_extension(mime_type) or ".bin"
    return f"upload{extension}"


def _file_to_data_url(path: str, *, mime_hint: str | None = None) -> str:
    mime_type = mime_hint or _guess_mime(path)
    payload = base64.b64encode(Path(path).read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{payload}"


def _normalize_match_values(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip().lower()] if value.strip() else []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            if isinstance(item, str) and item.strip():
                normalized.append(item.strip().lower())
        return normalized
    return []


def _provider_match_text(provider: Any, strategy: str | None) -> str:
    provider_config = getattr(provider, "provider_config", {}) or {}
    parts = [
        str(provider_config.get("id", "") or ""),
        str(provider_config.get("provider", "") or ""),
        str(provider_config.get("api_base", "") or ""),
        str(getattr(provider, "get_model", lambda: "")() or ""),
        str(strategy or ""),
    ]
    return " ".join(parts).lower()


def _provider_id(provider: Any) -> str:
    provider_config = getattr(provider, "provider_config", {}) or {}
    return str(provider_config.get("id", "") or "").strip()


def _provider_allowed(provider: Any, strategy: str | None, config: dict[str, Any]) -> bool:
    provider_text = _provider_match_text(provider, strategy)
    denylist = _normalize_match_values(config.get("provider_denylist"))
    if any(token in provider_text for token in denylist):
        return False
    allowlist = _normalize_match_values(config.get("provider_allowlist"))
    if allowlist and not any(token in provider_text for token in allowlist):
        return False
    return True


def _extract_openai_completion_text(completion: Any) -> str:
    choices = getattr(completion, "choices", None) or []
    if not choices:
        return ""
    message = getattr(choices[0], "message", None)
    if message is None:
        return ""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for item in content:
        if isinstance(item, str):
            text = item.strip()
            if text:
                parts.append(text)
            continue
        if isinstance(item, dict):
            if str(item.get("type", "")).lower() != "text":
                continue
            text_value = item.get("text")
            if isinstance(text_value, str):
                text = text_value.strip()
                if text:
                    parts.append(text)
            elif isinstance(text_value, dict):
                text = str(text_value.get("value", "") or "").strip()
                if text:
                    parts.append(text)
            continue
        text_value = getattr(item, "text", None)
        if isinstance(text_value, str):
            text = text_value.strip()
            if text:
                parts.append(text)
            continue
        nested_text = str(getattr(text_value, "value", "") or "").strip()
        if nested_text:
            parts.append(nested_text)
    return "\n".join(parts).strip()


def _read_usage_field(obj: Any, *names: str) -> Any:
    if obj is None:
        return None
    for name in names:
        if isinstance(obj, dict) and name in obj:
            return obj.get(name)
        value = getattr(obj, name, None)
        if value is not None:
            return value
    return None


def _coerce_usage_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


def _extract_token_usage_snapshot(response: Any) -> TokenUsageSnapshot | None:
    usage = _read_usage_field(response, "usage", "token_usage", "usage_metadata")
    if usage is None:
        raw_completion = _read_usage_field(response, "raw_completion")
        usage = _read_usage_field(raw_completion, "usage", "usage_metadata")
    if usage is None:
        usage = response

    prompt_details = _read_usage_field(usage, "prompt_tokens_details")
    cached_tokens = _coerce_usage_int(
        _read_usage_field(
            usage,
            "input_cached",
            "cached_tokens",
            "cache_read_input_tokens",
            "cached_content_token_count",
        )
    )
    if cached_tokens is None:
        cached_tokens = _coerce_usage_int(
            _read_usage_field(prompt_details, "cached_tokens")
        )

    input_tokens = _coerce_usage_int(
        _read_usage_field(
            usage,
            "input",
            "input_tokens",
            "prompt_tokens",
            "prompt_token_count",
        )
    )
    if input_tokens is None:
        input_other = _coerce_usage_int(_read_usage_field(usage, "input_other"))
        if input_other is not None:
            input_tokens = input_other + (cached_tokens or 0)

    output_tokens = _coerce_usage_int(
        _read_usage_field(
            usage,
            "output",
            "output_tokens",
            "completion_tokens",
            "candidates_token_count",
        )
    )
    total_tokens = _coerce_usage_int(
        _read_usage_field(usage, "total", "total_tokens", "total_token_count")
    )
    if total_tokens is None and (input_tokens is not None or output_tokens is not None):
        total_tokens = (input_tokens or 0) + (output_tokens or 0)

    if (
        input_tokens is None
        and output_tokens is None
        and total_tokens is None
        and cached_tokens is None
    ):
        return None
    return TokenUsageSnapshot(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
        cached_tokens=cached_tokens,
    )


def _is_direct_caption_provider(provider: Any) -> bool:
    return _provider_id(provider) == "__video_caption_direct__"


def _media_file_ref(media: SupportedMedia) -> str:
    if isinstance(media, Image):
        return str(media.url or media.file or "")
    return str(media.file or "")


def _path_without_query(value: str) -> str:
    if _is_http_url(value):
        return urlparse(value).path
    if value.startswith("file:///"):
        value = value[8:]
    return value


def _is_gif_ref(value: str) -> bool:
    if value.startswith("base64://"):
        return False
    return _path_without_query(value).lower().endswith(".gif")


def _is_supported_gif(component: Any, config: dict[str, Any]) -> bool:
    if not bool(config.get("enable_gif_input", False)):
        return False
    if not isinstance(component, Image):
        return False
    return _is_gif_ref(_media_file_ref(component))


def _is_gif_media(media: SupportedMedia) -> bool:
    return isinstance(media, Image) and _is_gif_ref(_media_file_ref(media))


def _media_mime_hint(media: SupportedMedia) -> str | None:
    if _is_gif_media(media):
        return "image/gif"
    return None


def _is_usable_media_ref(ref: str) -> bool:
    normalized = str(ref or "").strip()
    if not normalized:
        return False
    if normalized.startswith(("http://", "https://", "file:///", "base64://")):
        return True
    return os.path.exists(_normalize_local_file_path(normalized))


def _normalize_local_file_path(value: str) -> str:
    if value.startswith("file:///"):
        return value[8:]
    return value


def _extract_segments_from_onebot_payload(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict):
        data = payload.get("data")
        if isinstance(data, dict):
            payload = data
    segments = []
    if isinstance(payload, dict):
        segments = payload.get("message") or payload.get("messages") or []
    if not isinstance(segments, list):
        return []
    return [segment for segment in segments if isinstance(segment, dict)]


def _extract_onebot_video_entries(payload: Any) -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for segment in _extract_segments_from_onebot_payload(payload):
        if segment.get("type") != "video":
            continue
        data = segment.get("data") or {}
        if not isinstance(data, dict):
            continue
        entries.append(
            {
                "kind": "video",
                "file": str(data.get("file") or ""),
                "url": str(data.get("url") or ""),
                "path": str(data.get("path") or ""),
                "cover": str(data.get("cover") or ""),
                "file_id": str(data.get("file_id") or ""),
            }
        )
    return entries


def _extract_supported_media_from_chain(
    message_chain: list[Any],
    config: dict[str, Any],
) -> list[SupportedMedia]:
    media: list[SupportedMedia] = []
    for component in message_chain:
        if isinstance(component, Video):
            media.append(component)
        elif _is_supported_gif(component, config):
            media.append(component)
    return media


@dataclass
class CachedVideoMessage:
    session_id: str
    message_id: str
    videos: list[dict[str, str]]
    sender_id: str
    timestamp: int
    expires_at: int


@dataclass
class CachedVideoCaptionContext:
    session_id: str
    message_id: str
    caption_text: str
    remaining_rounds: int
    created_at: int


@dataclass
class CaptionAttemptResult:
    summary_text: str | None = None
    video_rejected: bool = False
    image_rejected: bool = False

    @property
    def blocks_native_video(self) -> bool:
        return self.video_rejected or self.image_rejected


@dataclass
class TokenUsageSnapshot:
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass
class MediaProbeInfo:
    duration_seconds: float | None = None
    width: int | None = None
    height: int | None = None
    fps: float | None = None
    frame_count: int | None = None


@dataclass
class FrameExtractionPlan:
    fps_expr: str
    frame_limit: int | None
    mode: str
    target_min_tokens: int | None = None
    target_max_tokens: int | None = None
    output_width: int | None = None
    output_height: int | None = None
    estimated_frame_tokens: int | None = None
    estimated_total_tokens: int | None = None

    def __iter__(self):
        yield self.fps_expr
        yield self.frame_limit


class DirectCaptionProvider:
    def __init__(
        self,
        *,
        provider_id: str,
        transport: str,
        api_base: str,
        api_key: str,
        model: str,
        timeout_seconds: int,
    ) -> None:
        normalized_transport = _normalize_video_transport(transport)
        normalized_api_base = _normalize_openai_base_url(api_base)
        normalized_model = _normalize_direct_caption_model(
            transport=normalized_transport,
            api_base=normalized_api_base,
            model=model,
        )
        self.provider_config = {
            "id": provider_id,
            "provider": "openai_chat_completion",
            "api_base": normalized_api_base,
            "model": normalized_model,
            "video_transport": normalized_transport,
            "model_metadata": {
                "modalities": {
                    "input": ["text", "image", "video"],
                    "output": ["text"],
                }
            },
        }
        self._api_key = api_key
        self._model = normalized_model
        self._timeout_seconds = max(10, int(timeout_seconds))

    def get_model(self) -> str:
        return self._model

    def get_current_key(self) -> str:
        return self._api_key

    async def text_chat(self, **kwargs):
        from openai import AsyncOpenAI

        default_headers = _kimi_compat_headers(self)
        client = AsyncOpenAI(
            api_key=self._api_key,
            base_url=str(self.provider_config.get("api_base", "") or ""),
            timeout=self._timeout_seconds,
            default_headers=default_headers or None,
        )
        completion = await client.chat.completions.create(
            model=self._model,
            messages=list(kwargs.get("contexts") or []),
            stream=False,
        )
        return SimpleNamespace(
            completion_text=_extract_openai_completion_text(completion),
            usage=_read_usage_field(completion, "usage"),
            raw_completion=completion,
        )


class VideoMessageCache:
    def __init__(self, ttl_seconds: int, max_entries: int) -> None:
        self.ttl_seconds = max(1, int(ttl_seconds))
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[tuple[str, str], CachedVideoMessage] = OrderedDict()

    def put(
        self,
        *,
        session_id: str,
        message_id: str,
        videos: list[dict[str, str]],
        sender_id: str,
        timestamp: int,
        now_ts: int | None = None,
    ) -> None:
        sid = session_id.strip()
        mid = _normalized_message_id(message_id)
        if not sid or not mid or not videos:
            return
        now = int(now_ts or time.time())
        key = (sid, mid)
        self._items[key] = CachedVideoMessage(
            session_id=sid,
            message_id=mid,
            videos=videos,
            sender_id=sender_id,
            timestamp=int(timestamp),
            expires_at=now + self.ttl_seconds,
        )
        self._items.move_to_end(key)
        self._prune(now)

    def get(
        self,
        *,
        session_id: str,
        message_id: str,
        now_ts: int | None = None,
    ) -> CachedVideoMessage | None:
        sid = session_id.strip()
        mid = _normalized_message_id(message_id)
        if not sid or not mid:
            return None
        now = int(now_ts or time.time())
        self._prune(now)
        key = (sid, mid)
        item = self._items.get(key)
        if not item:
            return None
        self._items.move_to_end(key)
        return item

    def _prune(self, now_ts: int | None = None) -> None:
        now = int(now_ts or time.time())
        expired_keys = [
            key for key, item in self._items.items() if int(item.expires_at) <= now
        ]
        for key in expired_keys:
            self._items.pop(key, None)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


class VideoCaptionContextCache:
    def __init__(self, max_entries: int) -> None:
        self.max_entries = max(1, int(max_entries))
        self._items: OrderedDict[tuple[str, str], CachedVideoCaptionContext] = OrderedDict()

    def put(
        self,
        *,
        session_id: str,
        message_id: str,
        caption_text: str,
        rounds: int,
        now_ts: int | None = None,
    ) -> None:
        sid = session_id.strip()
        mid = _normalized_message_id(message_id)
        text = caption_text.strip()
        keep_rounds = max(0, int(rounds))
        if not sid or not text or keep_rounds <= 0:
            return
        if not mid:
            mid = f"caption-{int((now_ts or time.time()) * 1000)}-{uuid.uuid4().hex}"
        key = (sid, mid)
        self._items[key] = CachedVideoCaptionContext(
            session_id=sid,
            message_id=mid,
            caption_text=text,
            remaining_rounds=keep_rounds,
            created_at=int(now_ts or time.time()),
        )
        self._items.move_to_end(key)
        self._prune()

    def consume(self, *, session_id: str) -> list[CachedVideoCaptionContext]:
        sid = session_id.strip()
        if not sid:
            return []
        self._prune()
        consumed: list[CachedVideoCaptionContext] = []
        for key in list(self._items.keys()):
            item = self._items.get(key)
            if item is None or item.session_id != sid:
                continue
            if item.remaining_rounds <= 0:
                self._items.pop(key, None)
                continue
            consumed.append(
                CachedVideoCaptionContext(
                    session_id=item.session_id,
                    message_id=item.message_id,
                    caption_text=item.caption_text,
                    remaining_rounds=item.remaining_rounds,
                    created_at=item.created_at,
                )
            )
            item.remaining_rounds -= 1
            if item.remaining_rounds <= 0:
                self._items.pop(key, None)
            else:
                self._items.move_to_end(key)
        return consumed

    def _prune(self) -> None:
        expired_keys = [
            key for key, item in self._items.items() if item.remaining_rounds <= 0
        ]
        for key in expired_keys:
            self._items.pop(key, None)
        while len(self._items) > self.max_entries:
            self._items.popitem(last=False)


class Main(Star):
    def __init__(self, context: Context, config: dict | None = None) -> None:
        super().__init__(context, config)
        merged = dict(DEFAULT_CONFIG)
        if isinstance(config, dict):
            self._ensure_config_defaults(config)
            merged.update(config)
        self.config = merged
        self.video_cache = VideoMessageCache(
            ttl_seconds=int(self.config.get("cache_ttl_seconds", 7200)),
            max_entries=int(self.config.get("cache_max_entries", 500)),
        )
        self.caption_context_cache = VideoCaptionContextCache(
            max_entries=int(self.config.get("cache_max_entries", 500)),
        )

    def _ensure_config_defaults(self, config: dict) -> None:
        updated = False
        for key, value in DEFAULT_CONFIG.items():
            if key not in config or config[key] is None:
                config[key] = value
                updated = True
        save_config = getattr(config, "save_config", None)
        if updated and callable(save_config):
            save_config()

    def _get_direct_caption_provider_config(self) -> dict[str, Any]:
        transport = _normalize_video_transport(
            self.config.get("video_caption_direct_transport", VIDEO_TRANSPORT_AUTO)
        )
        api_base = _normalize_openai_base_url(
            self.config.get("video_caption_direct_base_url", "")
        )
        return {
            "enabled": bool(self.config.get("video_caption_direct_enabled", False)),
            "transport": transport,
            "api_base": api_base,
            "api_key": str(
                self.config.get("video_caption_direct_api_key", "") or ""
            ).strip(),
            "model": _normalize_direct_caption_model(
                transport=transport,
                api_base=api_base,
                model=self.config.get("video_caption_direct_model", ""),
            ),
            "timeout_seconds": int(
                self.config.get("video_caption_direct_timeout_seconds", 120)
            ),
        }

    def _get_missing_direct_caption_config_keys(
        self,
        *,
        require_enabled: bool = True,
    ) -> list[str]:
        direct_config = self._get_direct_caption_provider_config()
        missing: list[str] = []
        if require_enabled and not direct_config["enabled"]:
            missing.append("video_caption_direct_enabled")
        for key in ("api_base", "api_key"):
            if not direct_config[key]:
                missing.append(f"video_caption_direct_{key}")
        if not direct_config["model"]:
            missing.append("video_caption_direct_model")
        return missing

    def _build_direct_caption_provider(
        self,
        *,
        require_enabled: bool = True,
    ) -> DirectCaptionProvider | None:
        missing_keys = self._get_missing_direct_caption_config_keys(
            require_enabled=require_enabled
        )
        if missing_keys:
            return None
        direct_config = self._get_direct_caption_provider_config()
        return DirectCaptionProvider(
            provider_id="__video_caption_direct__",
            transport=direct_config["transport"],
            api_base=direct_config["api_base"],
            api_key=direct_config["api_key"],
            model=direct_config["model"],
            timeout_seconds=direct_config["timeout_seconds"],
        )

    async def _run_direct_caption_connectivity_check(
        self,
        event: AstrMessageEvent,
    ) -> str:
        provider = self._build_direct_caption_provider(require_enabled=False)
        if provider is None:
            missing_keys = self._get_missing_direct_caption_config_keys(
                require_enabled=False
            )
            return "独立视频转述配置不完整：" + ", ".join(missing_keys)

        reply_id = self._find_reply_id(event)
        quoted_media = self._extract_quoted_media(event)
        quoted_media = await self._resolve_unusable_media_with_onebot(
            event,
            quoted_media,
            reply_id,
        )
        if quoted_media:
            result = await self._summarize_media_with_provider(
                media=quoted_media[:1],
                provider=provider,
                strategy=_detect_video_strategy(
                    provider,
                    mode="force",
                    prefer_model_metadata_video=True,
                ),
                user_question="请用一句话确认你看到了什么。",
            )
            if result.summary_text:
                return (
                    "独立视频转述模型测试成功。\n"
                    f"模型：{provider.get_model()}\n"
                    f"结果：{result.summary_text}"
                )
            if result.video_rejected and result.image_rejected:
                return (
                    "独立视频转述模型连通，但视频输入和抽帧图片输入都被拒绝。"
                )
            if result.video_rejected:
                return "独立视频转述模型连通，但当前接口拒绝视频输入。"
            return "独立视频转述模型已连通，但本次引用视频没有得到可用转述结果。"

        try:
            llm_resp = await provider.text_chat(
                contexts=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Reply with OK only.",
                            }
                        ],
                    }
                ]
            )
        except Exception as exc:  # noqa: BLE001
            return f"独立视频转述模型连通性检查失败：{exc}"

        completion_text = str(getattr(llm_resp, "completion_text", "") or "").strip()
        return (
            "独立视频转述模型文本连通性正常。"
            + (f"\n返回：{completion_text}" if completion_text else "")
            + "\n如需验证视频输入，请引用一条视频后再执行命令。"
        )

    @filter.command(
        "video_ref_test",
        alias={"测试视频转述模型", "测试视频模型"},
    )
    async def test_direct_video_caption_connectivity(
        self,
        event: AstrMessageEvent,
    ):
        result_text = await self._run_direct_caption_connectivity_check(event)
        yield event.plain_result(result_text)

    @filter.event_message_type(filter.EventMessageType.ALL, priority=10_000)
    async def capture_video_message(self, event: AstrMessageEvent) -> None:
        if not bool(self.config.get("enabled", True)):
            return
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return

        media = _extract_supported_media_from_chain(
            list(getattr(msg_obj, "message", None) or []),
            self.config,
        )
        if not media:
            return

        max_videos = max(1, int(self.config.get("max_videos_per_message", 3)))
        serialized_videos: list[dict[str, str]] = []
        for item in media[:max_videos]:
            serialized_videos.append(self._serialize_media(item))
        seen_refs = {
            f"{entry.get('file', '')}|{entry.get('url', '')}|{entry.get('path', '')}"
            for entry in serialized_videos
        }
        for entry in _extract_onebot_video_entries(getattr(msg_obj, "raw_message", None)):
            if len(serialized_videos) >= max_videos:
                break
            dedupe_key = (
                f"{entry.get('file', '')}|{entry.get('url', '')}|{entry.get('path', '')}"
            )
            if dedupe_key in seen_refs:
                continue
            ref = entry.get("url") or entry.get("file") or entry.get("path")
            if not ref:
                continue
            serialized_videos.append(entry)
            seen_refs.add(dedupe_key)

        if not serialized_videos:
            return

        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        message_id = _normalized_message_id(getattr(msg_obj, "message_id", ""))
        sender = getattr(msg_obj, "sender", None)
        sender_id = str(getattr(sender, "user_id", "") or "")
        timestamp = int(getattr(msg_obj, "timestamp", int(time.time())))
        self.video_cache.put(
            session_id=session_id,
            message_id=message_id,
            videos=serialized_videos,
            sender_id=sender_id,
            timestamp=timestamp,
        )

    @filter.on_llm_request(priority=10_000)
    async def inject_quoted_video(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if not bool(self.config.get("enabled", True)):
            return

        current_provider = self.context.get_using_provider(
            getattr(event, "unified_msg_origin", "")
        )
        if current_provider is None:
            return

        mode = str(self.config.get("mode", "auto") or "auto").lower()
        if mode == "off":
            return

        current_strategy = _detect_video_strategy(
            current_provider,
            mode=mode,
            prefer_model_metadata_video=bool(
                self.config.get("prefer_model_metadata_video", True)
            ),
        )
        if not _provider_allowed(current_provider, current_strategy, self.config):
            logger.info(
                "video-reference-vision: 已跳过，当前提供商被允许/拒绝列表过滤"
            )
            return

        if self._should_intercept_direct_video_request(event):
            logger.info(
                "video-reference-vision: 已拦截直接发送的非引用视频请求"
            )
            event.stop_event()
            return

        self._inject_cached_caption_context(event, req)

        reply_id = self._find_reply_id(event)
        quoted_media = self._extract_quoted_media(event, req=req)
        quoted_media = await self._resolve_unusable_media_with_onebot(
            event,
            quoted_media,
            reply_id,
        )
        if not quoted_media:
            return

        max_videos = max(1, int(self.config.get("max_videos_per_request", 1)))
        quoted_media = quoted_media[:max_videos]

        caption_provider, caption_strategy, using_current_caption_provider = (
            self._resolve_video_caption_provider(
                mode=mode,
                current_provider=current_provider,
                current_strategy=current_strategy,
            )
        )
        if caption_provider is not None:
            caption_result = await self._summarize_media_with_provider(
                media=quoted_media,
                provider=caption_provider,
                strategy=caption_strategy,
                user_question=self._extract_user_question(event, req),
            )
            if caption_result.summary_text:
                logger.info(
                    "video-reference-vision: 视频转述结果：%s",
                    caption_result.summary_text,
                )
                self._cache_caption_context(event, caption_result.summary_text)
                await self._rewrite_request_with_caption_text(
                    req,
                    caption_result.summary_text,
                )
                return
            if (
                using_current_caption_provider
                and caption_result.blocks_native_video
            ):
                if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                    req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                        list(req.extra_user_content_parts or [])
                    )
                logger.info(
                    "video-reference-vision: 当前提供商拒绝媒体转述输入，跳过原生视频注入"
                )
                return
            if _is_direct_caption_provider(caption_provider):
                if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                    req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                        list(req.extra_user_content_parts or [])
                    )
                logger.info(
                    "video-reference-vision: 独立转述提供商未生成摘要，跳过原生视频注入"
                )
                return

        if current_strategy is None:
            if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                    list(req.extra_user_content_parts or [])
                )
            logger.debug("video-reference-vision: 已跳过，当前提供商策略不匹配")
            return

        if _is_kimicode_provider(current_provider, strategy=current_strategy):
            if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                    list(req.extra_user_content_parts or [])
                )
            logger.info(
                "video-reference-vision: Kimi Code 传输方式跳过原生视频注入"
            )
            return

        if not bool(self.config.get("native_video_injection_fallback", True)):
            if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                    list(req.extra_user_content_parts or [])
                )
            logger.info(
                "video-reference-vision: 原生视频注入兜底已禁用"
            )
            return

        video_parts: list[dict] = []
        for item in quoted_media:
            part = await self._build_video_part(
                item,
                strategy=current_strategy,
                provider=current_provider,
            )
            if part:
                video_parts.append(part)

        if not video_parts:
            if str(self.config.get("fallback_behavior", "keep_text")) == "silent":
                req.extra_user_content_parts = _remove_video_attachment_text_from_extra_parts(
                    list(req.extra_user_content_parts or [])
                )
            return

        await self._rewrite_request_with_video_parts(req, video_parts)

    def _should_intercept_direct_video_request(self, event: AstrMessageEvent) -> bool:
        if bool(self.config.get("allow_direct_video", False)):
            return False
        if not bool(self.config.get("intercept_direct_video_llm_request", True)):
            return False

        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return False
        message_chain = list(getattr(msg_obj, "message", None) or [])
        if any(isinstance(comp, Reply) for comp in message_chain):
            return False
        return bool(_extract_supported_media_from_chain(message_chain, self.config))

    def _caption_context_cache_rounds(self) -> int:
        try:
            rounds = int(self.config.get("video_caption_context_cache_rounds", 0) or 0)
        except (TypeError, ValueError):
            return 0
        return max(0, min(rounds, 100))

    def _cache_caption_context(
        self,
        event: AstrMessageEvent,
        caption_text: str,
    ) -> None:
        rounds = self._caption_context_cache_rounds()
        if rounds <= 0:
            return
        msg_obj = getattr(event, "message_obj", None)
        message_id = _normalized_message_id(getattr(msg_obj, "message_id", ""))
        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        self.caption_context_cache.put(
            session_id=session_id,
            message_id=message_id,
            caption_text=caption_text,
            rounds=rounds,
        )
        logger.info(
            "video-reference-vision: 已缓存视频转述上下文，后续可保留 %d 轮",
            rounds,
        )

    def _inject_cached_caption_context(
        self,
        event: AstrMessageEvent,
        req: ProviderRequest,
    ) -> None:
        if self._caption_context_cache_rounds() <= 0:
            return
        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        cached_items = self.caption_context_cache.consume(session_id=session_id)
        if not cached_items:
            return
        extra_parts = list(req.extra_user_content_parts or [])
        extra_parts.append(TextPart(text=self._format_cached_caption_context(cached_items)))
        req.extra_user_content_parts = extra_parts
        logger.info(
            "video-reference-vision: 已注入缓存视频转述上下文：条数=%d",
            len(cached_items),
        )

    def _format_cached_caption_context(
        self,
        cached_items: list[CachedVideoCaptionContext],
    ) -> str:
        lines = [
            "[历史引用视频内容转述]",
            "以下是本会话最近引用过的视频转述内容，回答当前问题时可以参考。",
        ]
        for index, item in enumerate(cached_items, start=1):
            lines.append(f"视频 {index}：\n{item.caption_text}")
        return "\n\n".join(lines)

    def _resolve_video_caption_provider(
        self,
        *,
        mode: str,
        current_provider: Any | None = None,
        current_strategy: str | None = None,
    ) -> tuple[Any | None, str | None, bool]:
        direct_provider = self._build_direct_caption_provider(require_enabled=True)
        if direct_provider is not None:
            direct_strategy = _detect_video_strategy(
                direct_provider,
                mode=mode,
                prefer_model_metadata_video=True,
            )
            return direct_provider, direct_strategy, False
        if bool(self.config.get("video_caption_direct_enabled", False)):
            missing_keys = self._get_missing_direct_caption_config_keys(
                require_enabled=True
            )
            logger.warning(
                "video-reference-vision: 独立视频转述提供商配置不完整：%s",
                ", ".join(missing_keys),
            )

        provider_id = str(self.config.get("video_caption_provider_id", "") or "").strip()
        if not provider_id:
            if not bool(self.config.get("video_caption_use_current_provider", True)):
                return None, None, False
            if current_provider is None:
                return None, None, False
            return current_provider, current_strategy, True

        provider = self.context.get_provider_by_id(provider_id)
        if provider is None:
            logger.warning(
                "video-reference-vision: 未找到已配置的视频转述提供商：%s",
                provider_id,
            )
            return None, None, False

        strategy = _detect_video_strategy(
            provider,
            mode=mode,
            prefer_model_metadata_video=bool(
                self.config.get("prefer_model_metadata_video", True)
            ),
        )
        if strategy is None:
            logger.info(
                "video-reference-vision: 已配置的视频转述提供商不支持原生视频策略，如可用将依赖抽帧兜底：%s",
                provider_id,
            )
        if not _provider_allowed(provider, strategy, self.config):
            logger.info(
                "video-reference-vision: 已配置的视频转述提供商被允许/拒绝列表过滤：%s",
                provider_id,
            )
            return None, None, False
        current_provider_id = _provider_id(current_provider) if current_provider else ""
        using_current = bool(
            current_provider is not None
            and (
                provider is current_provider
                or (
                    current_provider_id
                    and current_provider_id == _provider_id(provider)
                )
            )
        )
        return provider, strategy, using_current

    def _find_reply_id(self, event: AstrMessageEvent) -> str:
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return ""
        for comp in list(getattr(msg_obj, "message", None) or []):
            if isinstance(comp, Reply):
                return _normalized_message_id(getattr(comp, "id", ""))
        return ""

    async def _resolve_onebot_video_entry(
        self,
        event: AstrMessageEvent,
        entry: dict[str, str],
    ) -> SupportedMedia | None:
        for key in ("url", "path"):
            ref = str(entry.get(key) or "").strip()
            if _is_usable_media_ref(ref):
                local_path = (
                    os.path.abspath(_normalize_local_file_path(ref))
                    if os.path.exists(_normalize_local_file_path(ref))
                    else ""
                )
                return Video(
                    file=ref,
                    cover=str(entry.get("cover") or ""),
                    path=local_path,
                )

        file_ref = str(entry.get("file") or entry.get("file_id") or "").strip()
        if not file_ref:
            return None

        bot = getattr(event, "bot", None)
        if bot is None:
            return None

        try:
            result = await bot.call_action("get_file", file=file_ref)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "video-reference-vision: OneBot get_file 获取 %s 失败：%s",
                file_ref,
                exc,
            )
            return None

        if not isinstance(result, dict):
            return None

        resolved_ref = (
            str(result.get("url") or "").strip()
            or str(result.get("file") or "").strip()
            or str(result.get("path") or "").strip()
        )
        if not resolved_ref:
            return None

        local_path = ""
        normalized = _normalize_local_file_path(resolved_ref)
        if os.path.exists(normalized):
            local_path = os.path.abspath(normalized)

        return Video(
            file=resolved_ref,
            cover=str(entry.get("cover") or ""),
            path=local_path,
        )

    async def _resolve_quoted_media_from_onebot_get_msg(
        self,
        event: AstrMessageEvent,
        reply_id: str,
    ) -> list[SupportedMedia]:
        if not bool(self.config.get("enable_onebot_media_resolver", True)):
            return []

        bot = getattr(event, "bot", None)
        if bot is None or not reply_id:
            return []

        try:
            payload = await bot.call_action("get_msg", message_id=int(reply_id))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "video-reference-vision: OneBot get_msg 获取 reply_id=%s 失败：%s",
                reply_id,
                exc,
            )
            return []

        media: list[SupportedMedia] = []
        for entry in _extract_onebot_video_entries(payload):
            resolved = await self._resolve_onebot_video_entry(event, entry)
            if resolved:
                media.append(resolved)

        if media:
            logger.info(
                "video-reference-vision: 已通过 OneBot get_msg/get_file 解析到 %d 个引用视频",
                len(media),
            )
        return media

    async def _resolve_unusable_media_with_onebot(
        self,
        event: AstrMessageEvent,
        media: list[SupportedMedia],
        reply_id: str,
    ) -> list[SupportedMedia]:
        usable_media: list[SupportedMedia] = []
        has_unusable = False

        for item in media:
            ref = _media_file_ref(item)
            if _is_usable_media_ref(ref):
                usable_media.append(item)
            else:
                has_unusable = True

        if media and not has_unusable:
            return usable_media

        onebot_media = await self._resolve_quoted_media_from_onebot_get_msg(
            event,
            reply_id,
        )
        if onebot_media:
            return onebot_media

        return media

    async def _resolve_media_local_path(self, media: SupportedMedia) -> str:
        file_ref = _media_file_ref(media).strip()
        path_ref = str(getattr(media, "path", "") or "").strip()

        try:
            return await media.convert_to_file_path()
        except Exception as exc:  # noqa: BLE001
            for candidate in (path_ref, file_ref):
                normalized = _normalize_local_file_path(candidate.strip())
                if normalized and os.path.exists(normalized):
                    logger.info(
                        "video-reference-vision: convert_to_file_path 失败，回退到 media.path/本地文件：%s",
                        normalized,
                    )
                    return os.path.abspath(normalized)

            raise FileNotFoundError(
                f"unable to resolve media to local path: file={file_ref!r}, path={path_ref!r}"
            ) from exc

    def _extract_user_question(self, event: AstrMessageEvent, req: ProviderRequest) -> str:
        prompt = str(req.prompt or "").strip()
        if prompt:
            return prompt
        return str(getattr(event, "message_str", "") or "").strip()

    def _build_video_caption_prompt(self, user_question: str) -> str:
        prompt_text = str(self.config.get("video_caption_prompt", "") or "").strip()
        if not prompt_text:
            prompt_text = DEFAULT_VIDEO_CAPTION_PROMPT
        if bool(self.config.get("video_caption_use_current_question", True)) and user_question:
            prompt_text = f"{prompt_text}\n\n用户当前问题：{user_question}"
        return prompt_text

    def _looks_like_rejected_media_error(self, exc: Exception, *, media_kind: str) -> bool:
        text = str(exc or "").lower()
        if media_kind == "video":
            tokens = (
                "video_url",
                "input video",
                "support input video",
                "unsupported video",
                "does not support video",
                "does not support 'video_url'",
                "supported types: ['text', 'image']",
                'supported types: ["text", "image"]',
            )
            return any(token in text for token in tokens)
        if media_kind == "image":
            tokens = (
                "image_url",
                "input image",
                "support input image",
                "unsupported image",
                "does not support image",
                "supported types: ['text']",
                'supported types: ["text"]',
            )
            return any(token in text for token in tokens)
        return False

    def _looks_like_token_limit_error(self, exc: Exception) -> bool:
        text = str(exc or "").lower()
        tokens = (
            "token limit",
            "context length",
            "maximum context",
            "max context",
            "too many tokens",
            "exceeded model token",
            "exceeded token",
            "请求超过",
            "上下文",
        )
        return any(token in text for token in tokens)

    def _get_ffmpeg_command(self) -> str:
        return str(self.config.get("ffmpeg_path", "") or "").strip() or "ffmpeg"

    def _get_ffprobe_command(self) -> str:
        return str(self.config.get("ffprobe_path", "") or "").strip() or "ffprobe"

    @staticmethod
    def _parse_frame_rate(value: Any) -> float | None:
        text = str(value or "").strip()
        if not text or text == "0/0":
            return None
        if "/" in text:
            numerator, denominator = text.split("/", 1)
            try:
                denominator_value = float(denominator)
                if denominator_value == 0:
                    return None
                value = float(numerator) / denominator_value
            except ValueError:
                return None
        else:
            try:
                value = float(text)
            except ValueError:
                return None
        return value if value > 0 else None

    @staticmethod
    def _optional_positive_float(value: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    @staticmethod
    def _optional_positive_int(value: Any) -> int | None:
        try:
            number = int(float(value))
        except (TypeError, ValueError):
            return None
        return number if number > 0 else None

    def _probe_media_info(self, local_path: str) -> MediaProbeInfo:
        ffprobe_cmd = self._get_ffprobe_command()
        if shutil.which(ffprobe_cmd) is None and not os.path.exists(ffprobe_cmd):
            return MediaProbeInfo()
        try:
            proc = subprocess.run(
                [
                    ffprobe_cmd,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height,avg_frame_rate,r_frame_rate,nb_frames,duration:format=duration",
                    "-of",
                    "json",
                    local_path,
                ],
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception:
            return MediaProbeInfo()
        try:
            payload = json.loads(proc.stdout or "{}")
        except json.JSONDecodeError:
            return MediaProbeInfo()

        streams = payload.get("streams") or []
        stream = streams[0] if streams else {}
        duration_seconds = self._optional_positive_float(stream.get("duration"))
        if duration_seconds is None:
            duration_seconds = self._optional_positive_float(
                (payload.get("format") or {}).get("duration")
            )
        fps = self._parse_frame_rate(stream.get("avg_frame_rate"))
        if fps is None:
            fps = self._parse_frame_rate(stream.get("r_frame_rate"))
        frame_count = self._optional_positive_int(stream.get("nb_frames"))
        if frame_count is None and duration_seconds and fps:
            frame_count = max(1, math.ceil(duration_seconds * fps))
        return MediaProbeInfo(
            duration_seconds=duration_seconds,
            width=self._optional_positive_int(stream.get("width")),
            height=self._optional_positive_int(stream.get("height")),
            fps=fps,
            frame_count=frame_count,
        )

    def _probe_media_duration_seconds(self, local_path: str) -> float | None:
        return self._probe_media_info(local_path).duration_seconds

    def _extract_frame_data_urls_sync(
        self,
        *,
        local_path: str,
    ) -> list[str]:
        ffmpeg_cmd = self._get_ffmpeg_command()
        if shutil.which(ffmpeg_cmd) is None and not os.path.exists(ffmpeg_cmd):
            raise FileNotFoundError(f"未找到 ffmpeg：{ffmpeg_cmd}")

        probe_info = self._probe_media_info(local_path)
        plan = self._resolve_frame_extraction_params(
            duration_seconds=probe_info.duration_seconds,
            probe_info=probe_info,
        )
        self._log_frame_extraction_plan(local_path, probe_info, plan)

        with tempfile.TemporaryDirectory(prefix="video_ref_frames_") as tmp_dir:
            output_pattern = os.path.join(tmp_dir, "frame_%03d.jpg")
            command = [
                ffmpeg_cmd,
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                local_path,
                "-vf",
                self._build_ffmpeg_frame_filter(plan),
            ]
            if plan.frame_limit is not None:
                command.extend(["-frames:v", str(plan.frame_limit)])
            command.extend(["-q:v", "5"])
            command.append(output_pattern)
            subprocess.run(command, check=True, capture_output=True, text=True)
            frame_files = sorted(glob.glob(os.path.join(tmp_dir, "frame_*.jpg")))
            frame_refs = [
                _file_to_data_url(path, mime_hint="image/jpeg") for path in frame_files
            ]
            actual_tokens = self._estimate_data_url_tokens(frame_refs)
            actual_size_bytes = sum(len(item.encode("utf-8")) for item in frame_refs)
            logger.info(
                "video-reference-vision: 抽帧完成：实际抽帧=%d，data URL 总大小=%.2fMB，按请求体保守估算 token≈%s；临时帧目录已自动清理",
                len(frame_refs),
                actual_size_bytes / 1024 / 1024,
                self._format_optional_int(actual_tokens),
            )
            return frame_refs

    def _resolve_frame_extraction_params(
        self,
        *,
        duration_seconds: float | None,
        probe_info: MediaProbeInfo | None = None,
    ) -> FrameExtractionPlan:
        frame_mode = str(
            self.config.get("video_caption_frame_mode", FRAME_MODE_AUTO) or FRAME_MODE_AUTO
        ).strip().lower()
        if frame_mode == FRAME_MODE_AUTO:
            return self._resolve_auto_frame_extraction_params(
                duration_seconds=duration_seconds,
                probe_info=probe_info or MediaProbeInfo(duration_seconds=duration_seconds),
            )

        if frame_mode == FRAME_MODE_FPS:
            frame_fps = max(
                0.1,
                float(self.config.get("video_caption_frame_fps", 1.0) or 1.0),
            )
            frame_limit: int | None = None
            if duration_seconds and duration_seconds > 0:
                frame_limit = max(1, math.ceil(duration_seconds * frame_fps))
            return FrameExtractionPlan(
                fps_expr=f"{frame_fps:.6f}",
                frame_limit=frame_limit,
                mode=FRAME_MODE_FPS,
            )

        frame_count = max(1, int(self.config.get("video_caption_frame_count", 4)))
        fps_expr = "1"
        if duration_seconds and duration_seconds > 0:
            fps_value = max(frame_count / max(duration_seconds, 1.0), 0.001)
            fps_expr = f"{fps_value:.6f}"
        return FrameExtractionPlan(
            fps_expr=fps_expr,
            frame_limit=frame_count,
            mode=FRAME_MODE_COUNT,
        )

    def _resolve_auto_frame_extraction_params(
        self,
        *,
        duration_seconds: float | None,
        probe_info: MediaProbeInfo,
    ) -> FrameExtractionPlan:
        min_context_tokens = self._config_context_tokens(
            "video_caption_frame_auto_min_context_k",
            150,
        )
        max_context_tokens = self._config_context_tokens(
            "video_caption_frame_auto_max_context_k",
            200,
        )
        if min_context_tokens > max_context_tokens:
            min_context_tokens, max_context_tokens = (
                max_context_tokens,
                min_context_tokens,
            )

        budget_for_frames = max(max_context_tokens - AUTO_FRAME_TEXT_TOKENS, 1000)
        scaled_probe_info = self._scale_probe_to_token_budget(
            probe_info,
            target_frame_tokens=max(AUTO_FRAME_MIN_FRAME_TOKENS, budget_for_frames),
        )
        estimated_frame_tokens = self._estimate_frame_tokens(scaled_probe_info)
        frame_count = max(1, budget_for_frames // max(estimated_frame_tokens, 1))
        if probe_info.frame_count:
            frame_count = min(frame_count, max(1, probe_info.frame_count))

        fps_expr = "1"
        if duration_seconds and duration_seconds > 0:
            fps_value = max(frame_count / max(duration_seconds, 1.0), 0.001)
            if probe_info.fps:
                fps_value = min(fps_value, probe_info.fps)
            fps_expr = f"{fps_value:.6f}"

        return FrameExtractionPlan(
            fps_expr=fps_expr,
            frame_limit=frame_count,
            mode=FRAME_MODE_AUTO,
            target_min_tokens=min_context_tokens,
            target_max_tokens=max_context_tokens,
            output_width=(
                scaled_probe_info.width
                if scaled_probe_info.width != probe_info.width
                or scaled_probe_info.height != probe_info.height
                else None
            ),
            output_height=(
                scaled_probe_info.height
                if scaled_probe_info.width != probe_info.width
                or scaled_probe_info.height != probe_info.height
                else None
            ),
            estimated_frame_tokens=estimated_frame_tokens,
            estimated_total_tokens=(estimated_frame_tokens * frame_count)
            + AUTO_FRAME_TEXT_TOKENS,
        )

    def _scale_probe_to_token_budget(
        self,
        probe_info: MediaProbeInfo,
        *,
        target_frame_tokens: int,
    ) -> MediaProbeInfo:
        if not probe_info.width or not probe_info.height:
            return probe_info
        current_tokens = self._estimate_frame_tokens(probe_info)
        if current_tokens <= target_frame_tokens:
            return probe_info

        scale = math.sqrt(max(target_frame_tokens, 1) / max(current_tokens, 1))
        width = self._even_dimension(math.floor(probe_info.width * scale))
        height = self._even_dimension(math.floor(probe_info.height * scale))
        width = min(width, probe_info.width)
        height = min(height, probe_info.height)

        scaled_probe_info = MediaProbeInfo(
            duration_seconds=probe_info.duration_seconds,
            width=width,
            height=height,
            fps=probe_info.fps,
            frame_count=probe_info.frame_count,
        )
        while (
            width > 2
            and height > 2
            and self._estimate_frame_tokens(scaled_probe_info) > target_frame_tokens
        ):
            width = self._even_dimension(math.floor(width * 0.95))
            height = self._even_dimension(math.floor(height * 0.95))
            scaled_probe_info.width = width
            scaled_probe_info.height = height
        return scaled_probe_info

    @staticmethod
    def _even_dimension(value: int) -> int:
        dimension = max(2, int(value))
        if dimension % 2:
            dimension -= 1
        return max(2, dimension)

    @staticmethod
    def _build_ffmpeg_frame_filter(plan: FrameExtractionPlan) -> str:
        filters = [f"fps={plan.fps_expr}"]
        if plan.output_width and plan.output_height:
            filters.append(
                f"scale={plan.output_width}:{plan.output_height}:flags=lanczos"
            )
        return ",".join(filters)

    def _config_context_tokens(self, key: str, default_k: int) -> int:
        try:
            value_k = int(float(self.config.get(key, default_k) or default_k))
        except (TypeError, ValueError):
            value_k = default_k
        return max(1, value_k) * 1000

    def _estimate_frame_tokens(self, probe_info: MediaProbeInfo) -> int:
        if not probe_info.width or not probe_info.height:
            return AUTO_FRAME_DEFAULT_TOKENS
        vision_tiles = math.ceil(probe_info.width / 512) * math.ceil(
            probe_info.height / 512
        )
        vision_tokens = 85 + (170 * max(1, vision_tiles))
        return max(AUTO_FRAME_DEFAULT_TOKENS, vision_tokens)

    @staticmethod
    def _estimate_data_url_tokens(data_urls: list[str]) -> int:
        return math.ceil(
            sum(len(item) for item in data_urls) / AUTO_FRAME_BASE64_CHARS_PER_TOKEN
        )

    @staticmethod
    def _format_optional_float(value: float | None, *, suffix: str = "") -> str:
        if value is None:
            return "未知"
        return f"{value:.2f}{suffix}"

    @staticmethod
    def _format_optional_int(value: int | None, *, suffix: str = "") -> str:
        if value is None:
            return "未知"
        return f"{value}{suffix}"

    @staticmethod
    def _format_resolution(probe_info: MediaProbeInfo) -> str:
        if not probe_info.width or not probe_info.height:
            return "未知"
        return f"{probe_info.width}x{probe_info.height}"

    @staticmethod
    def _format_output_resolution(plan: FrameExtractionPlan) -> str:
        if not plan.output_width or not plan.output_height:
            return "保持原分辨率"
        return f"{plan.output_width}x{plan.output_height}"

    def _log_frame_extraction_plan(
        self,
        local_path: str,
        probe_info: MediaProbeInfo,
        plan: FrameExtractionPlan,
    ) -> None:
        logger.info(
            "video-reference-vision: 视频信息：文件=%s，分辨率=%s，帧率=%s，时长=%s，总帧数=%s",
            os.path.basename(local_path),
            self._format_resolution(probe_info),
            self._format_optional_float(probe_info.fps, suffix="fps"),
            self._format_optional_float(probe_info.duration_seconds, suffix="秒"),
            self._format_optional_int(probe_info.frame_count),
        )
        logger.info(
            "video-reference-vision: 抽帧计划：模式=%s，抽帧频率=%sfps，最多抽帧=%s，输出分辨率=%s，目标上下文=%s-%s token，预计单帧≈%s token，预计总消耗≈%s token",
            plan.mode,
            plan.fps_expr,
            self._format_optional_int(plan.frame_limit),
            self._format_output_resolution(plan),
            self._format_optional_int(plan.target_min_tokens),
            self._format_optional_int(plan.target_max_tokens),
            self._format_optional_int(plan.estimated_frame_tokens),
            self._format_optional_int(plan.estimated_total_tokens),
        )

    def _log_api_token_usage(
        self,
        label: str,
        llm_resp: Any,
        *,
        frame_count: int | None = None,
    ) -> None:
        usage = _extract_token_usage_snapshot(llm_resp)
        frame_text = (
            f"，帧数={frame_count}"
            if frame_count is not None
            else ""
        )
        if usage is None:
            logger.info(
                "video-reference-vision: %s API 实际 token 用量%s：提供商未返回 usage",
                label,
                frame_text,
            )
            return
        logger.info(
            "video-reference-vision: %s API 实际 token 用量%s：输入=%s，输出=%s，总计=%s，缓存输入=%s",
            label,
            frame_text,
            self._format_optional_int(usage.input_tokens),
            self._format_optional_int(usage.output_tokens),
            self._format_optional_int(usage.total_tokens),
            self._format_optional_int(usage.cached_tokens),
        )

    async def _extract_frame_data_urls(self, media: SupportedMedia) -> list[str]:
        refs: list[str] = []
        if isinstance(media, Video):
            cover_ref = str(media.cover or "").strip()
            if _is_usable_media_ref(cover_ref):
                refs.append(cover_ref)

        try:
            local_path = await self._resolve_media_local_path(media)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "video-reference-vision: 为抽帧兜底解析本地媒体路径失败：%s",
                exc,
            )
            return refs

        try:
            frame_refs = await asyncio.to_thread(
                self._extract_frame_data_urls_sync,
                local_path=local_path,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "video-reference-vision: 抽帧失败：%s",
                exc,
            )
            return refs

        for item in frame_refs:
            if item not in refs:
                refs.append(item)
        return refs

    async def _build_frame_caption_blocks(
        self,
        *,
        media: list[SupportedMedia],
    ) -> list[dict[str, Any]]:
        blocks: list[dict[str, Any]] = []
        for media_index, item in enumerate(media, start=1):
            frame_refs = await self._extract_frame_data_urls(item)
            for frame_index, frame_ref in enumerate(frame_refs, start=1):
                blocks.append(
                    ImageURLPart(
                        image_url=ImageURLPart.ImageURL(
                            url=frame_ref,
                            id=f"video_{media_index}_frame_{frame_index}",
                        )
                    ).model_dump()
                )
        return blocks

    @staticmethod
    def _select_evenly_spaced_blocks(
        blocks: list[dict[str, Any]],
        target_count: int,
    ) -> list[dict[str, Any]]:
        if target_count >= len(blocks):
            return list(blocks)
        if target_count <= 1:
            return [blocks[0]]
        selected: list[dict[str, Any]] = []
        used_indices: set[int] = set()
        last_index = len(blocks) - 1
        for step in range(target_count):
            index = round(step * last_index / (target_count - 1))
            if index in used_indices:
                continue
            selected.append(blocks[index])
            used_indices.add(index)
        return selected

    @staticmethod
    def _build_frame_caption_contexts(
        *,
        prompt_text: str,
        frame_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"{prompt_text}\n\n"
                            "如果原生视频输入不可用，请根据这些抽帧图片推断视频内容。"
                        ),
                    },
                    *frame_blocks,
                ],
            }
        ]

    def _build_reduced_frame_blocks_for_retry(
        self,
        frame_blocks: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        target_count = max(1, len(frame_blocks) // 2)
        return self._select_evenly_spaced_blocks(frame_blocks, target_count)

    async def _summarize_media_with_provider(
        self,
        *,
        media: list[SupportedMedia],
        provider: Any,
        strategy: str | None,
        user_question: str,
    ) -> CaptionAttemptResult:
        result = CaptionAttemptResult()
        prefer_frame_caption = _is_kimicode_provider(provider, strategy=strategy)

        if prefer_frame_caption:
            logger.info(
                "video-reference-vision: Kimi Code 传输方式当前不支持原生视频解析，优先使用抽帧转述"
            )
        elif strategy:
            video_parts: list[dict] = []
            for item in media:
                part = await self._build_video_part(
                    item,
                    strategy=strategy,
                    provider=provider,
                )
                if part:
                    video_parts.append(part)
            if video_parts:
                contexts = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self._build_video_caption_prompt(user_question)},
                            *video_parts,
                        ],
                    }
                ]
                try:
                    llm_resp = await provider.text_chat(contexts=contexts)
                except Exception as exc:  # noqa: BLE001
                    if self._looks_like_rejected_media_error(exc, media_kind="video"):
                        result.video_rejected = True
                    logger.warning(
                        "video-reference-vision: 视频转述请求失败：%s",
                        exc,
                    )
                else:
                    self._log_api_token_usage("视频转述", llm_resp)
                    summary = str(getattr(llm_resp, "completion_text", "") or "").strip()
                    if summary:
                        result.summary_text = summary
                        return result
                    logger.warning(
                        "video-reference-vision: 视频转述提供商返回了空文本"
                    )

        if not prefer_frame_caption and not bool(
            self.config.get("video_caption_frame_fallback", True)
        ):
            return result

        frame_blocks = await self._build_frame_caption_blocks(media=media)
        if not frame_blocks:
            return result

        prompt_text = self._build_video_caption_prompt(user_question)
        current_frame_blocks = frame_blocks
        attempt = 1
        while True:
            contexts = self._build_frame_caption_contexts(
                prompt_text=prompt_text,
                frame_blocks=current_frame_blocks,
            )
            try:
                llm_resp = await provider.text_chat(contexts=contexts)
            except Exception as exc:  # noqa: BLE001
                if (
                    self._looks_like_token_limit_error(exc)
                    and len(current_frame_blocks) > 1
                ):
                    reduced_frame_blocks = self._build_reduced_frame_blocks_for_retry(
                        current_frame_blocks
                    )
                    if len(reduced_frame_blocks) < len(current_frame_blocks):
                        logger.warning(
                            "video-reference-vision: 抽帧转述触发 token 上限，自动减少帧数后重试：第 %d 次失败，当前帧数=%d，下次帧数=%d，错误=%s",
                            attempt,
                            len(current_frame_blocks),
                            len(reduced_frame_blocks),
                            exc,
                        )
                        current_frame_blocks = reduced_frame_blocks
                        attempt += 1
                        continue
                if self._looks_like_rejected_media_error(exc, media_kind="image"):
                    result.image_rejected = True
                logger.warning(
                    "video-reference-vision: 抽帧转述请求失败：%s",
                    exc,
                )
                return result
            break

        self._log_api_token_usage(
            "抽帧转述",
            llm_resp,
            frame_count=len(current_frame_blocks),
        )
        summary = str(getattr(llm_resp, "completion_text", "") or "").strip()
        if not summary:
            logger.warning("video-reference-vision: 抽帧转述提供商返回了空文本")
            return result
        result.summary_text = summary
        return result

    def _extract_quoted_media(
        self,
        event: AstrMessageEvent,
        *,
        req: ProviderRequest | None = None,
    ) -> list[SupportedMedia]:
        msg_obj = getattr(event, "message_obj", None)
        if msg_obj is None:
            return []

        message_chain = list(getattr(msg_obj, "message", None) or [])
        reply_comp: Reply | None = None
        for comp in message_chain:
            if isinstance(comp, Reply):
                reply_comp = comp
                break

        if reply_comp is None:
            if bool(self.config.get("allow_direct_video", False)):
                return _extract_supported_media_from_chain(message_chain, self.config)
            return []

        if reply_comp.chain:
            media = _extract_supported_media_from_chain(list(reply_comp.chain), self.config)
            if media:
                return media

        reply_id = _normalized_message_id(getattr(reply_comp, "id", ""))
        if not reply_id:
            return []

        session_id = str(getattr(event, "unified_msg_origin", "") or "")
        cached = self.video_cache.get(session_id=session_id, message_id=reply_id)
        hydrated: list[SupportedMedia] = []
        if cached:
            for entry in cached.videos:
                media = self._hydrate_media(entry)
                if media:
                    hydrated.append(media)
        if hydrated:
            return hydrated
        if req is not None:
            return self._extract_videos_from_request_attachment_text(req)
        return []

    def _serialize_media(self, media: SupportedMedia) -> dict[str, str]:
        if isinstance(media, Image):
            return {
                "kind": "gif",
                "file": str(media.file or ""),
                "url": str(media.url or ""),
                "path": str(media.path or ""),
        }
        file_ref = str(media.file or "")
        return {
            "kind": "video",
            "file": file_ref,
            "url": file_ref if _is_http_url(file_ref) else "",
            "cover": str(media.cover or ""),
            "path": str(media.path or ""),
            "file_id": "",
        }

    def _hydrate_media(self, entry: dict[str, str]) -> SupportedMedia | None:
        kind = str(entry.get("kind", "video") or "video")
        file_ref = str(entry.get("file", "") or "")
        path_ref = str(entry.get("path", "") or "")
        url_ref = str(entry.get("url", "") or "")
        if kind == "gif":
            ref = url_ref or file_ref or path_ref
            if not ref:
                return None
            return Image(file=ref, url=url_ref, path=path_ref)

        ref = url_ref or file_ref or path_ref
        if not ref:
            return None
        return Video(
            file=ref,
            cover=str(entry.get("cover", "") or ""),
            path=path_ref,
        )

    def _extract_videos_from_request_attachment_text(
        self,
        req: ProviderRequest,
    ) -> list[SupportedMedia]:
        videos: list[SupportedMedia] = []
        max_videos = max(1, int(self.config.get("max_videos_per_request", 1)))
        for part in list(req.extra_user_content_parts or []):
            text = getattr(part, "text", None)
            if not isinstance(text, str):
                continue
            path = _extract_path_from_video_attachment_text(text)
            if not path:
                continue
            videos.append(Video(file=path, path=path))
            if len(videos) >= max_videos:
                break
        return videos

    def _resolve_kimi_part_mode(
        self,
        *,
        strategy: str,
        size_bytes: int | None,
        provider: Any | None = None,
    ) -> str:
        kimi_strategy = str(self.config.get("kimi_strategy", "auto") or "auto").lower()
        if strategy != "kimi":
            return "base64"
        if kimi_strategy == "upload":
            return "upload"
        if kimi_strategy == "base64":
            return "base64"
        provider_config = getattr(provider, "provider_config", {}) or {}
        if _uses_kimicode_transport(
            transport=_provider_video_transport(provider),
            api_base=provider_config.get("api_base", ""),
        ):
            return "upload"
        max_size_mb = max(1, int(self.config.get("max_base64_mb", 20)))
        if size_bytes is None:
            return "base64"
        if size_bytes > max_size_mb * 1024 * 1024 and bool(
            self.config.get("kimi_upload_on_oversize", True)
        ):
            return "upload"
        return "base64"

    async def _build_video_part(
        self,
        media: SupportedMedia,
        *,
        strategy: str,
        provider: Any,
    ) -> dict | None:
        file_ref = _media_file_ref(media)
        prefer_public_url = bool(self.config.get("prefer_public_url", True))
        kimi_part_mode = self._resolve_kimi_part_mode(
            strategy=strategy,
            size_bytes=None,
            provider=provider,
        )
        mime_hint = _media_mime_hint(media)
        allow_kimi_upload = not _is_gif_media(media)
        kimi_upload_supported = allow_kimi_upload and _supports_kimi_file_upload(provider)

        if (
            strategy == "kimi"
            and kimi_part_mode == "upload"
            and allow_kimi_upload
            and not kimi_upload_supported
        ):
            logger.info(
                "video-reference-vision: 当前 Kimi 兼容端点未开放文件上传，回退到本地 base64"
            )

        if (
            strategy == "kimi"
            and kimi_part_mode == "upload"
            and kimi_upload_supported
        ):
            try:
                local_path = await self._resolve_media_local_path(media)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "video-reference-vision: 为 Kimi 上传解析本地媒体路径失败：%s",
                    exc,
                )
                return None
            logger.info("video-reference-vision: 使用 Kimi 上传模式")
            return await self._build_kimi_upload_video_part(
                provider=provider,
                local_path=local_path,
            )

        if prefer_public_url and _is_http_url(file_ref) and strategy != "kimi":
            logger.info("video-reference-vision: 使用公网媒体 URL")
            video_url = file_ref
        else:
            try:
                local_path = await self._resolve_media_local_path(media)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "video-reference-vision: 解析本地媒体路径失败：%s",
                    exc,
                )
                return None

            size_bytes = os.path.getsize(local_path)
            max_size_mb = max(1, int(self.config.get("max_base64_mb", 20)))
            if size_bytes > max_size_mb * 1024 * 1024 and (
                strategy != "kimi" or not kimi_upload_supported
            ):
                logger.warning(
                    "video-reference-vision: 已跳过超大本地媒体（%.2fMB > %dMB）：%s",
                    size_bytes / 1024 / 1024,
                    max_size_mb,
                    local_path,
                )
                return None

            if (
                strategy == "kimi"
                and kimi_upload_supported
                and self._resolve_kimi_part_mode(
                    strategy=strategy,
                    size_bytes=size_bytes,
                    provider=provider,
                )
                == "upload"
            ):
                logger.info("video-reference-vision: 使用 Kimi 上传模式")
                return await self._build_kimi_upload_video_part(
                    provider=provider,
                    local_path=local_path,
                )

            logger.info("video-reference-vision: 使用本地媒体 base64 载荷")
            video_url = _file_to_data_url(local_path, mime_hint=mime_hint)

        part = {"type": "video_url", "video_url": {"url": video_url}}
        if strategy == "qwen":
            part["fps"] = float(self.config.get("qwen_fps", 2.0))
        elif strategy in {"openrouter", "generic"}:
            part["fps"] = float(self.config.get("generic_fps", 2.0))
        return part

    async def _build_kimi_upload_video_part(
        self,
        *,
        provider: Any,
        local_path: str,
    ) -> dict | None:
        try:
            from openai import AsyncOpenAI

            provider_config = getattr(provider, "provider_config", {}) or {}
            direct_api_base = _normalize_openai_base_url(provider_config.get("api_base"))
            configured_api_base = _normalize_openai_base_url(
                self.config.get("kimi_api_base")
            )
            if _is_direct_caption_provider(provider):
                api_base = direct_api_base or configured_api_base or "https://api.moonshot.cn/v1"
            else:
                api_base = configured_api_base or direct_api_base or "https://api.moonshot.cn/v1"
            api_key = str(getattr(provider, "get_current_key", lambda: "")() or "")
            if not api_key:
                logger.warning("video-reference-vision: Kimi 上传缺少 API Key")
                return None
            default_headers = _kimi_compat_headers(provider)
            client = AsyncOpenAI(
                api_key=api_key,
                base_url=api_base,
                default_headers=default_headers or None,
            )
            mime_type = _guess_mime(local_path)
            file_obj = await client.post(
                "/files",
                cast_to=KimiFileObject,
                body={"purpose": "video"},
                files={
                    "file": (
                        _guess_kimi_upload_filename(mime_type),
                        Path(local_path).read_bytes(),
                        mime_type,
                    )
                },
                options={"headers": {"Content-Type": "multipart/form-data"}},
            )
            file_id = str(getattr(file_obj, "id", "") or "")
            if not file_id:
                logger.warning("video-reference-vision: Kimi 上传返回了空文件 ID")
                return None
            return {"type": "video_url", "video_url": {"url": f"ms://{file_id}"}}
        except Exception as exc:  # noqa: BLE001
            logger.warning("video-reference-vision: Kimi 上传失败：%s", exc)
            return None

    async def _rewrite_request_with_video_parts(
        self,
        req: ProviderRequest,
        video_parts: list[dict],
    ) -> None:
        extra_parts = list(req.extra_user_content_parts or [])
        if bool(self.config.get("remove_default_video_text", True)):
            extra_parts = _remove_video_attachment_text_from_extra_parts(extra_parts)
        normalized_parts: list[Any] = []
        for part in video_parts:
            if (
                isinstance(part, dict)
                and part.get("type") == "video_url"
                and isinstance(part.get("video_url"), dict)
            ):
                normalized_parts.append(VideoURLPart.model_validate(part))
            else:
                normalized_parts.append(part)
        req.image_urls = []
        req.audio_urls = []
        req.extra_user_content_parts = extra_parts + normalized_parts

    async def _rewrite_request_with_caption_text(
        self,
        req: ProviderRequest,
        caption_text: str,
    ) -> None:
        extra_parts = list(req.extra_user_content_parts or [])
        if bool(self.config.get("remove_default_video_text", True)):
            extra_parts = _remove_video_attachment_text_from_extra_parts(extra_parts)
        extra_parts.append(TextPart(text=f"[引用视频内容转述]\n{caption_text}"))
        req.image_urls = []
        req.audio_urls = []
        req.extra_user_content_parts = extra_parts
