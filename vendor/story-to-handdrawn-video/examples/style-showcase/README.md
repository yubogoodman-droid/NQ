# 75 种风格 · 45 秒翻书演示

https://github.com/user-attachments/assets/9c7ebd73-16fe-4428-8d2c-99006469b201

[下载 1080p 配乐视频](https://github.com/gnipbao/story-to-handdrawn-video/raw/refs/heads/main/examples/style-showcase/handdrawn-styles-75-page-flip-45s-bgm.mp4)

上方为 720×960 轻量预览，保留完整 45 秒内容和配乐。下载文件为 1080×1440 原片。

选用项目统一生成的“老人和小孩牵手过石拱桥”配图，75 个不同风格覆盖全部 11 类，按画材交错排列。原图完整置入，右下角卷页露出下一张图片；平均约 0.6 秒一页，首尾稍作停留。

- 成片：45 秒，1350 帧，1080 × 1440，30 fps，H.264 / yuv420p / BT.709。
- 配乐：Carefree — Kevin MacLeod；AAC 192 kbps / 48 kHz 立体声。
- 音频处理：原曲前 45 秒，0.7 秒淡入、2.5 秒淡出；实际响度 −18.04 LUFS，真峰值 −3.68 dBTP。
- 检查：75 页代表帧及卷页画面检查、音视频完整解码、时长与帧数验证通过。加配乐时直接复制画面码流，视频 SHA-256 与静音成片一致。

[风格顺序与时间表（CSV）](style-sequence.csv) · [详细时间线（JSON）](style-sequence.json) · [验证记录](validation.json)

配图位置在项目根目录的 `references/style-examples/standard/`，选择与时间线按 `style-sequence.json` 的 source、start_frame、duration_frames 和 curl_frames 记录。演示使用 Skill 的完整图片导入与翻页渲染，再以 FFmpeg 添加背景音乐；Skill 默认视频输出仍为静音。

## 音乐署名

音乐采用 CC BY 4.0。分享本配乐视频时，请保留 [BGM_CREDITS.md](BGM_CREDITS.md) 中的署名及许可链接。
