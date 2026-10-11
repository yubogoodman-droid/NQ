# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- 327 typed assets (297 style entries and 30 palettes), expanding the original
  20 with 277 additional recipes. Original selectors remain compatible;
  additional recipe IDs use HS-001 through HS-277.
- Searchable offline catalog and CLI filters, with 30 editorially curated
  defaults, medium categories, and explicit palette selection.
- Agent-driven reference-image analysis, checksummed profiles and private
  saved styles with copied references (`custom:<id>`). No automatic public
  propagation of user images or profiles.
- Isolated catalog, profile, planning, FFmpeg import and transition regressions.
- 327 local previews of one grandmother-and-child bridge scene: 20 existing
  project comparisons and 307 newly generated images, with exact prompts,
  individual generation records and SHA-256 checksums.
- Offline thumbnail enlargement and a concise README with standard comparisons.
- A 45-second, 75-style page-flip showcase with post-production BGM, a timed
  style list, verified export metadata, and Creative Commons music attribution.

### Changed

- Generated handwritten caption images are the default; no automatic Ma Shan
  Zheng font loading. System-font fallback is explicit and direct-cut only.
- One shared post-processor for generated and uploaded pages; page flips keep
  untouched masters. Palette/profile changes create fresh asset batches.
- Generated-page crops detect the actual caption/art gutter instead of assuming
  image models honor pixel coordinates; ambiguous separation fails explicitly.
- Reference-profile negative traits use explicit `Avoid:` constraints.
- Gallery previews use project artwork throughout. Project references appear at
  the end of the README; text-data provenance and licenses remain intact.
- Offline checks separate storyboard schema checks from strict asset checks;
  each render validates its own storyboard assets. Historical generated images
  are still excluded from the repository.
- Page-curl containers remain transparent outside the folded page, so the
  incoming page is visible during the transition instead of a white gap.
- Local audio caches are excluded from Git and source sharing.

### Evidence and rollback

- Asset action: merge existing recipes/workflows, add reference-style lifecycle.
  Local deterministic paths are covered by executed fixtures. Real generated
  lettering and both transitions are tracked in `examples/replay-report.md`;
  one replay does not establish quality across all 297 recipes.
- All 327 standard previews decode successfully and match their recorded hashes;
  generation scope, gallery verification and limitations are recorded in
  `examples/standard-preview-report.md`.
- Development baseline is fbab5b2. Revert the scoped update to roll back;
  user runtime style directories and local audio caches remain separate.

## [1.1.0] - 2026-08-08

### Added

- Built-in library of 20 selectable hand-drawn styles with aliases, intended
  use cases, prompt recipes, negative constraints, and source attribution.
- One square comparison sample per style plus a labeled contact sheet.
- `--style` selection by order, id, Chinese name, English name, or alias,
  and `--list-styles` catalog output with example paths.

### Changed

- Keep the approved colored-pencil diary look as the immutable default while
  allowing non-default styles to use independent prompt locks and palettes.
- Include the selected style and reference fingerprint in generated asset
  caches so switching styles cannot silently reuse stale images.
- Expand the Skill contract, UI metadata, and bilingual README with visual
  selection guidance and the complete style table.

## [1.0.0] - 2026-07-21

### Added

- Initial open-source release of `story-to-handdrawn-video`.
- Remotion renderer (`src/`): story-text beats, uploaded-page cropping,
  left-to-right `text → bw → color` reveals, page-flip transition,
  safe contained framing, silent MP4 output (final 1080×1440, preview 720×960).
- Unified CLI entry `scripts/run_story_video.py` with plan / generate /
  import / render / preview / full modes and Codex Image2 + OpenAI API generators.
- Distributable agent skill in `skill-package/story-to-handdrawn-video/`
  with `STORY_VIDEO_PROJECT` discovery and upward directory walk.
- Example story, style references, and the Ma Shan Zheng font (OFL).

[Unreleased]: https://github.com/gnipbao/story-to-handdrawn-video/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/gnipbao/story-to-handdrawn-video/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/gnipbao/story-to-handdrawn-video/releases/tag/v1.0.0
