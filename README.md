# 给清漪做的引用视频理解

> 注意：目前本插件的视频转述模型里，只有 Moonshot 官方 API（`api.moonshot.ai` / `api.moonshot.cn`）支持优先尝试原生视频解析；Kimi Code（`api.kimi.com/coding/v1`）当前只能稳定走抽帧转述，不再尝试原生 `video_url` 视频解析；其他平台/代理网关（如 OpenAI-compatible 聚合网关）暂未适配视频输入。
> 作者主页：https://github.com/Sisyphbaous-DT-Project

给 AstrBot 用的一个小插件：用户引用一条视频消息提问时，插件会尽量把“被引用的视频内容”变成模型真正能理解的输入，而不是只把 AstrBot 默认的附件提示文本发过去。

它不是“群里一发视频就自动分析”的插件。只有用户引用视频并明确提问时，插件才会介入。

## 现在的真实工作方式

当前版本的默认链路已经不是“无脑把 `video_url` 塞给当前聊天模型”了，而是按下面顺序处理：

1. 先找到被引用的视频。
2. 优先走“视频转述”链路。
3. 转述成功后，只把转述文本回写给 AstrBot 主对话。
4. 只有转述失败且允许回退时，才继续尝试原生 `video_url` 注入。

这样做的原因很直接：很多 OpenAI-compatible 网关、代理层或回退模型并不真正支持 `video_url`，硬塞过去只会把整条请求打失败。

## 它解决的问题

AstrBot 能识别视频消息，也会生成类似下面这样的附件提示：

```text
[Video Attachment: name demo.mp4, path D:\xxx\demo.mp4]
```

但这段文本本身不等于“模型读到了视频内容”。模型大概率只知道“这里有个视频文件”，并不知道视频里发生了什么。

这个插件的目标是：

- 在用户引用视频提问时，真正恢复出被引用的视频资源。
- 尽量让模型读取视频内容本身，或者至少读取关键帧。
- 最终把视频内容变成对当前问题有用的文本输入。

## 当前支持的解析链路

插件为了兼容 QQ / OneBot / NapCat 这类场景，已经做了多层兜底：

1. 优先从 `Reply.chain` 里直接取 `Video`。
2. 如果引用链不完整，就用 `Reply.id` 去插件缓存里找。
3. 如果缓存里拿到的是无效 `file`，会尝试通过 OneBot/NapCat 的 `get_msg` 回查原始消息。
4. 如果原始 `video` 段里只有资源 ID，还会继续调用 `get_file` 解析成真实 URL 或本地文件。
5. 如果前面都没拿到，再尝试从 AstrBot 默认生成的 `[Video Attachment ...]` 文本里反解析本地路径。

这也是它在 QQ 引用视频场景里比“只读 `Reply.chain`”更稳的原因。

## 默认转述链路

拿到视频后，插件会优先尝试把视频内容“转述成文本”。

默认策略如下：

1. 如果配置了 `video_caption_provider_id`，优先用这个模型做视频转述。
2. 如果没配置，并且 `video_caption_use_current_provider=true`，就先用当前聊天模型做转述。
3. 如果配置的是 `kimicode`，会直接跳过原生 `video_url`，优先用 `ffmpeg` 抽关键帧，再按 `image_url` 发给转述模型。
4. 其他支持原生视频输入的链路，才会先尝试直接发 `video_url`。
5. 如果视频输入被网关拒绝，并且 `video_caption_frame_fallback=true`，就继续改走抽帧图片输入。
6. 转述成功后，把结果写成一段文本，例如：

```text
[引用视频内容转述]
……
```

然后只把这段文本喂回 AstrBot 主对话。

这个模式对“当前聊天模型不支持视频，但支持图片”的场景尤其有用。

## 原生视频注入现在是什么角色

原生 `video_url` 注入没有被删除，但它现在是回退路径，不再是首选路径。

只有在这些条件满足时，它才会继续尝试：

- 视频转述链路没有成功返回文本。
- 当前 provider 能匹配到可用的视频策略。
- `native_video_injection_fallback=true`。

所以如果你看到日志里出现“当前 provider 拒绝视频输入，跳过原生注入”，这通常是插件在避免把整条请求打崩，而不是插件失效了。

## `ffmpeg` 抽帧兜底

当前版本已经实现了 `ffmpeg` 抽帧兜底。

行为是：

- 当视频原生输入被拒绝时，插件会尝试把本地视频抽成若干关键帧。
- 再把这些关键帧按 `image_url` 发给转述模型。
- 如果模型支持图片但不支持视频，这条链通常还能工作。

注意：

- 这一步需要环境里能找到 `ffmpeg`；没有就会自动跳过。
- `ffprobe` 可选；有的话会用来估算视频时长并更均匀地抽帧。

### `ffmpeg` 安装说明

- 插件不会内置 `ffmpeg` / `ffprobe`，需要用户自行安装。
- 推荐把 `ffmpeg` 与 `ffprobe` 加到系统 `PATH`，插件会自动发现。
- 如果没有加到 `PATH`，请在插件配置里填写 `ffmpeg_path` 与 `ffprobe_path` 的绝对路径。
- 当日志出现 `未找到 ffmpeg` 时，表示当前环境未正确安装或路径未配置。

## 使用方式

推荐用法：

1. 先在聊天里发一条视频。
2. 再引用这条视频。
3. 问一个具体问题。

示例：

```text
用户：<发了一条视频>

用户：<引用刚才的视频>
这段视频里的人在做什么？
```

插件会尽量把这条引用视频变成：

- 视频转述文本，或
- 视频关键帧输入，或
- 最后再回退到原生 `video_url`

而不是只把“这里有个视频附件”交给模型。

## 普通发视频会发生什么

插件会缓存视频消息的信息，方便后面引用时找回来。

缓存阶段不会：

- 自动分析视频
- 上传视频
- 强行触发视频注入

如果你没有引用视频，只是单独发了一条视频：

- 插件默认不会主动帮你分析
- 如果 `allow_direct_video=false` 且 `intercept_direct_video_llm_request=true`，插件还会拦截这类“非引用的纯视频请求”

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

注意：视频抽帧与图片兜底能力依赖本机 `ffmpeg`，请先完成安装再测试视频转述链路。

## 关键配置项

### 基础行为

- `enabled`：是否启用插件。
- `mode`：建议先用 `auto`；`force` 会更激进地匹配视频能力，调试时再用。
- `fallback_behavior`：失败时怎么处理。默认 `keep_text`，也就是保留 AstrBot 原来的附件文本提示；`silent` 会尽量移除它。
- `allow_direct_video`：是否允许“当前消息直接带视频”时也注入视频。默认关闭。
- `intercept_direct_video_llm_request`：不允许直发视频时，是否拦截这类请求。

### QQ / OneBot / NapCat 相关

- `enable_onebot_media_resolver`：当引用视频只有资源 ID、文件名或不完整引用链时，尝试通过 OneBot/NapCat 的 `get_msg` / `get_file` 回查真实资源。
- `cache_ttl_seconds`：视频缓存保留多久。
- `cache_max_entries`：全局缓存上限。

### 转述链路

- `video_caption_provider_id`：单独指定视频转述模型。留空时会优先用当前聊天模型做转述。
- `video_caption_use_current_provider`：未指定转述模型时，是否先用当前聊天模型尝试转述。
- `video_caption_direct_enabled`：启用插件内独立视频转述通道。
- `video_caption_direct_transport`：独立转述通道的视频传输方式。`moonshot` 会优先尝试原生视频；`kimicode` 当前会直接走抽帧转述；`generic` 按普通 OpenAI-compatible 多模态处理，`auto` 自动判断。
- `video_caption_direct_base_url` / `video_caption_direct_api_key` / `video_caption_direct_model`：独立转述通道自己的接口地址、密钥和模型，不影响 AstrBot 主聊天模型。选择 `kimicode` 时，插件会按 Kimi Code 官方约定自动使用 `kimi-for-coding`，模型输入框可留空或忽略。
- `video_caption_direct_test_entry`：设置页里显式显示测试命令入口。当前 AstrBot WebUI 还不支持插件在这里直接挂一个自定义动作按钮，所以这里会直接提示使用 `/video_ref_test`。
- `video_caption_prompt`：发给视频转述模型的提示词。
- `video_caption_use_current_question`：转述时是否带上用户当前问题。

### 抽帧兜底

- `video_caption_frame_fallback`：视频原生输入失败时，是否用 `ffmpeg` 抽帧后改走图片输入。选择 `kimicode` 时，抽帧会被提升到第一优先级。
- `video_caption_frame_mode`：抽帧模式。`auto` 表示按视频时长、分辨率、帧率和上下文预算自动抽帧，并在必要时自动缩小抽帧分辨率；`count` 表示按总帧数均匀抽帧；`fps` 表示按每秒抽帧数量抽帧。
- `video_caption_frame_count`：最多发多少张关键帧。
- `video_caption_frame_fps`：仅在 `fps` 模式下生效，表示每秒抽多少帧。
- `video_caption_frame_auto_min_context_k`：仅在 `auto` 模式下生效，自动抽帧的目标最小上下文，单位为 k token，默认 `150`。
- `video_caption_frame_auto_max_context_k`：仅在 `auto` 模式下生效，自动抽帧的目标最大上下文，单位为 k token，默认 `200`。建议不要超过 `200`，给 Kimi 262k 上限留出余量。
- `ffmpeg_path`：可选，手动指定 `ffmpeg` 路径。
- `ffprobe_path`：可选，手动指定 `ffprobe` 路径。

自动抽帧会在日志中输出视频分辨率、帧率、时长、总帧数、计划抽帧数、输出分辨率、预计 token 消耗和实际抽帧后的 data URL token 估算，方便判断是否接近模型上下文上限。如果抽帧转述请求仍然顶到 token 上限，插件会自动减少帧数并重试。

### 原生视频注入

- `native_video_injection_fallback`：转述链路失败后，是否继续回退到原生 `video_url` 注入。
- `prefer_public_url`：视频本身是公网 URL 时，优先直接传 URL。
- `max_base64_mb`：本地视频转 base64 的大小上限。
- `remove_default_video_text`：成功注入后，是否移除默认的 `[Video Attachment ...]` 文本。

### Kimi 相关

- `kimi_strategy=auto`：自动决定走 base64 还是上传。
- `kimi_strategy=upload`：强制先上传，再用 `ms://file_id` 引用。
- `kimi_strategy=base64`：强制本地转 base64。
- `kimi_upload_on_oversize=true`：视频超过 base64 限制时，自动改走上传模式。
- `kimi_api_base`：可选覆盖 Kimi 接口地址。
- `kimicode` 当前不再尝试插件内原生视频解析，而是直接复用抽帧转述链路。

### GIF

- `enable_gif_input`：把被引用的 GIF 按完整动图处理，而不是只看第一帧。

## 当前已经实现

- 引用视频触发，不主动分析所有视频。
- 视频消息缓存，支持按 `Reply.id` 回查。
- QQ / OneBot / NapCat 场景下通过 `get_msg` / `get_file` 回查真实视频资源。
- 本地路径失效时的安全兜底，不再因裸文件名直接抛错。
- 视频转述模型链路。
- 未单独指定转述模型时，优先使用当前聊天模型尝试转述。
- 视频原生输入失败时，自动抽帧并改走图片输入。
- 转述成功后，把结果回写成文本，不再继续把 `video_url` 传给主对话。
- 必要时再回退到原生 `video_url` 注入。
- Qwen / DashScope 的 `video_url` 注入。
- OpenRouter / 通用 OpenAI-compatible 的 `video_url` 注入。
- Kimi 上传模式入口。

## 目前仍不做的事

这些不在当前版本目标里：

- 收到视频后立刻自动分析。
- 自动做音轨转写。
- 给所有厂商做完整私有协议适配。
- 改 AstrBot core，让所有 provider 都原生理解视频能力。

## 排查问题

如果引用视频后仍然不符合预期，优先看这些：

- 当前聊天模型或转述模型是否真的支持视频，或者至少支持图片。
- 日志里是否出现：
  - `已通过 OneBot get_msg/get_file 解析到 ... 个引用视频`
  - `视频转述请求失败`
  - `抽帧转述请求失败`
  - `当前提供商拒绝媒体转述输入，跳过原生视频注入`
- 环境里是否有可用的 `ffmpeg`。
- 视频是否超过 `max_base64_mb`。
- QQ 引用链是否丢失，且缓存是否已经过期。

如果日志显示：

- 视频输入被拒绝，但随后抽帧成功并生成转述文本：这是正常降级。
- 视频和图片都被当前 provider 拒绝：说明这条接入本身不适合做视频理解，建议单独配置一个视频转述模型。
