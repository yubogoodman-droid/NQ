# 主题资产库

内置 **327 项资产**：297 条画风配方与 30 种主题配色。默认显示 30 种常用精选，可按类型、画材、题材、名称和编号查询。

## 统一配图

全部示例使用「老人和小孩牵手走过石拱桥」场景：老人居左、小孩居右，完整石拱桥下有少量水纹，两侧少量竹叶，方形画幅与白色留白。不同画风按自己的线条、造型、材质和色彩重新绘制；单色画风允许用明暗区分衣服。配色示例使用同一种水粉画材和同一场景，只改变主题色。

打开 [本地目录](style-library.html)，点击示意图可站内放大。默认 30 种按常见用途选编；完整目录保留全部 327 项。示例不代替具体故事的逐字校对和构图检查。

## 调用

```bash
npm run styles
npm run styles -- --all
npm run styles -- --category watercolor
npm run styles -- --type palette
npm run styles -- --type all --json
python3 scripts/run_story_video.py --input examples/story.txt --style HS-128 --palette C-01 --mode generate
```

1–20 的编号、ID、中文名及别名保持兼容。扩展风格使用 `HS-001`…`HS-277`，菜单序号为 21–297。名称重复时使用 ID。`C-01`…`C-30` 只改变显式指定的主题配色，保留所选画材。

## 分类与默认精选

画风类别：日记与生活、线描与讲解、蜡笔与彩铅、水墨与国风、水彩与淡彩、水粉与绘本、版画与印刷、拼贴与纸艺、几何与平面、漫画与幽默、动画与材质。配色按蓝、绿、古典红绿、粉紫、暖阳大地分组。

默认 30 项包含原有 20 项，以及城市细线小景 HS-003、水彩手写生活札记 HS-021、松线淡彩绘本 HS-036、极简诗意绘本 HS-045、低饱和冷幽默绘本 HS-047、中古几何童话 HS-061、极简时尚线描 HS-085、透明留白水彩 HS-088、四季旅行水彩日记 HS-128、肌理剪纸拼贴绘本 HS-225。

## 参考图与个人风格

提供故事和 `--style-reference /absolute/reference.png` 后，Agent 看图分析线条、造型、配色、材质、构图、文字六维特征，写出 profile 并继续生成。纯 CLI 只准备分析任务，不会自行执行视觉模型。

提出收藏时，用 `--style-profile /absolute/profile.json --save-style my-style` 保存；随后 `--style custom:my-style` 复用。参考图随 profile 复制到项目 `.story-video/styles/`，可通过 `STORY_VIDEO_STYLE_HOME` 指定存储目录；个人资产进入完整菜单、分类和搜索，不改变公共 30 项精选，也不进入 Git 或分享包。

## 维护

- 标准图片位于 `style-examples/standard/`，配图清单记录场景版本、实际生成工具、提示词和文件校验值。
- `npm run styles:build` 重建机器目录与网页，`npm run check:styles` 校验可复现性及配图完整性。
- 标准场景只用于配图比较。生成新故事时，人物、动作和场景仍以当前用户的故事为准，不能把过桥人物带入新故事。
- 修改风格应更新配方与对应示例，保留 ID；项目参考与许可证集中见 [README 文末](../README.md#参考与许可)。
