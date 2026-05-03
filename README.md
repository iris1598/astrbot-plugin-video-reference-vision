# 引用视频理解

给 AstrBot 用的一个小插件：用户引用一条视频消息提问时，把这条视频按 `video_url` 的形式塞进本次 LLM 请求里，让支持视频输入的模型真的去看视频。

它不是“群里一发视频就自动分析”的插件。这个设计是故意的：大多数视频只是聊天内容，没必要每条都上传、转 base64 或消耗模型 token。只有用户引用视频并明确提问时，插件才会介入。

## 它解决的问题

AstrBot 现在能识别视频消息，也能把视频下载到本地。但在发给模型时，视频通常只会变成一段文本提示，例如：

```text
[Video Attachment: name demo.mp4, path D:\xxx\demo.mp4]
```

模型看到的是“这里有个视频文件”，不是视频内容本身。这个插件做的事情就是：在用户引用视频提问时，把这类附件提示替换成真正的多模态视频输入。

目前主要面向 OpenAI-compatible 的多模态接口，优先兼容：

- Qwen / DashScope 视频模型
- OpenRouter 上支持视频输入的模型
- Kimi 新版视觉模型
- 其他明确支持 `video_url` 的接口

纯文本模型、只支持图片不支持视频的模型，不会因为装了这个插件就突然能看视频。

## 使用方式

推荐用法很简单：

1. 先在聊天里发一条视频。
2. 再引用这条视频，问一个具体问题。
3. 插件会尝试找到被引用的视频，并把它注入给当前模型。

示例：

```text
用户：<发了一条视频>

用户：<引用刚才的视频>
这段视频里的人在做什么？
```

如果模型支持视频输入，它应该会根据视频内容回答，而不是只说“看到一个视频附件”。

## 普通发视频会发生什么

插件会缓存视频消息的信息，方便后面引用时找回来。缓存阶段不会上传视频，也不会把视频转成 base64。

需要注意的是：插件本身不会对“直接发视频”做视频注入；至于 AstrBot 主流程是否会因为纯视频消息触发一次默认 LLM 请求，取决于你当前的 AstrBot 配置和唤醒逻辑。如果你发现“只发视频也会被机器人回复”，那是 AstrBot 原本的视频附件文本流程在工作，不是本插件在识别视频。

后续版本会考虑加一个开关，用来在插件缓存完视频后直接拦截这类纯视频 LLM 请求。

## 插件怎么找被引用的视频

引用消息在不同平台上的表现不太一样，尤其是 QQ / OneBot v11：有时引用链很完整，有时只给一个 reply id。所以插件做了几层兜底：

1. 优先从 `Reply.chain` 里直接取 `Video`。
2. 如果引用链不完整，就用 `Reply.id` 去插件缓存里找。
3. 如果缓存也没命中，再尝试从 AstrBot 默认生成的 `[Video Attachment ...]` 文本里反解析本地路径。

这也是为什么这个插件对 QQ 场景会比“只读 Reply.chain”的实现更稳一些。

## 安装

把仓库放到 AstrBot 的插件目录：

```text
AstrBot/data/plugins/astrbot_plugin_video_reference_vision
```

目录里至少需要这些文件：

```text
main.py
metadata.yaml
_conf_schema.json
README.md
```

重载插件后，在 WebUI 里确认插件已经启用。

## 配置项说明

默认配置可以先不动。常用项大概是这些：

- `enabled`：是否启用插件。
- `mode`：建议先用 `auto`；`force` 会强行尝试注入，调试时再用。
- `prefer_public_url`：视频本身是公网 URL 时，优先直接传 URL。
- `max_base64_mb`：本地视频转 base64 的大小上限，默认 20MB。
- `qwen_fps`：Qwen / DashScope 的视频抽帧频率，默认 2.0。
- `generic_fps`：通用 OpenAI-compatible 模式下使用的 fps。
- `fallback_behavior`：注入失败时怎么处理。默认 `keep_text`，也就是保留 AstrBot 原来的附件文本提示。
- `allow_direct_video`：是否允许“当前消息直接带视频”时也注入视频。默认关闭，推荐保持关闭。

Kimi 相关：

- `kimi_strategy=auto`：自动选择策略。
- `kimi_strategy=upload`：上传视频后用 `ms://file_id` 引用。
- `kimi_strategy=base64`：直接把视频转 base64。
- `kimi_upload_on_oversize=true`：视频超过 base64 限制时，自动尝试 Kimi 上传模式。

## 当前已经实现

- 引用视频触发，不主动把每条视频都喂给模型。
- 视频消息缓存，支持通过 `Reply.id` 回查。
- 从 `Reply.chain` 直接读取视频。
- 从 `[Video Attachment ...]` 文本里兜底解析本地路径。
- Qwen / DashScope 的 `video_url` 注入。
- OpenRouter / 通用 OpenAI-compatible 的 `video_url` 注入。
- Kimi 上传模式入口。
- 注入成功后移除默认视频附件文本，避免模型同时看到“真实视频”和“本地路径提示”。

## 目前不做的事

这些不在当前版本目标里：

- 收到视频后立刻自动分析。
- ffmpeg 抽帧兜底。
- 音轨转写。
- 给所有厂商做完整适配。
- 把视频能力改进 AstrBot core。

现在先把“引用视频提问”这条链路跑稳，后面的事再慢慢补。

## 排查问题

引用视频后，如果模型还是只回答“有一个视频附件”，优先检查这些：

- 当前模型是否真的支持视频输入。
- 插件是否启用。
- 视频是否超过 `max_base64_mb`。
- 当前 provider 是否是 OpenAI-compatible 接口。
- 日志里是否有“跳过处理”“视频过大”“Kimi 上传失败”等提示。
- QQ 引用链是否丢失，且视频缓存是否已经过期。

## 一句话总结

这个插件只做一件事：当你引用一条视频并提问时，尽量让 AstrBot 把视频本体交给支持视频理解的模型，而不是只给模型看一段本地路径文本。
