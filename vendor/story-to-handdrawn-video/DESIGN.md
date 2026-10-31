# Rendering contract

## Pipeline

The Agent Skill and renderer have separate responsibilities:

1. The Agent reads the story, preserves its wording, plans sentence-based beats, and establishes character continuity. A selected built-in recipe or an analyzed reference-image profile provides the visual style.
2. The available image-generation tool creates a character reference and complete master pages. In the default `image2` text mode, exact handwritten captions and illustrations are generated together. Inspect glyphs, character identity, margins, and reference fidelity before import.
3. Local scripts import the approved masters and prepare the assets required by the selected animation mode. Uploaded pages enter at this step and follow the same processing rules.
4. Remotion reads the storyboard, applies timing and animation, and renders a silent H.264 video. Voiceover and background music are optional post-production work.

Preparing a generation manifest is not image generation. The default CLI path needs the Agent to call an image tool and supply real output files before import and render can complete.

## Canvas

- Final output: 1080×1440, 30fps, white background
- Preview output: 720×960, same aspect ratio and timing
- Captions stay in the upper safe area
- Illustrations use `object-fit: contain`; never crop with `cover`

## Motion

### Layered reveal (`cut`)

- The default generated page has a caption above the illustration. Uploaded composite pages use the same layout analysis.
- Separate the caption and illustration at the observed content boundary, preserving their proportions with contain/pad fitting.
- Derive the black-and-white plate from the final color plate locally so both share the same canvas and pixel alignment.
- Reveal `text → bw_full → color`, from left to right. This is a layer reveal, not a reconstruction of individual drawing strokes.
- A page that cannot be split reliably needs corrected artwork or an appropriate full-page layout; do not crop through text or characters to fit a nominal coordinate.

### Page flip (`page-flip`)

- Display the complete, untouched master as a static page, followed by a bottom-right page curl.
- Preserve existing captions, color and composition. Do not add a second caption, split the page, or introduce black-and-white/recoloring stages.
- The underside carries a faded version of the original page; the next page is visible beneath the curl.
- Adjacent pages overlap for the transition. Rendered duration is the sum of page frame counts minus the overlap between adjacent pages; use the storyboard timing calculation when targeting an exact runtime.

Both modes avoid camera shake and bounce. The renderer does not add narration or text-synchronized music.

## Assets

- Generated and uploaded masters share `scripts/page-assets.mjs`; uploaded masters are copied into a content-addressed generated directory.
- Keep the complete master as the source. Derived caption plates and illustration plates have their own fitted canvases; black-and-white and color illustration plates are aligned with each other.
- Text defaults to image-tool generated handwriting. Correct wrong glyphs in the master before importing; never silently replace them with a font. The explicit `font` mode is a separate, user-selected fallback.
- Selected recipes, reference-image hashes and palette overrides affect the generated-asset fingerprint, preventing stale batches from being reused after a style change.
- Generated assets and rendered videos are runtime outputs. New checkouts do not include historical generated batches; strict storyboard validation requires the referenced assets to be imported first.

## Reference styles and privacy

The Agent inspects reference images and records six dimensions: line, shape, palette, material, composition and lettering. The CLI prepares the analysis request and validates the resulting profile; it does not perform visual analysis by itself.

A reference profile guides new story artwork. People, wording and actions depicted in the reference do not automatically become story content. Character continuity remains controlled by the story and its character reference.

When the user asks to save a style, store the profile and copied reference images in the project's ignored `.story-video/styles/` directory (or the configured private style home). Saved styles can be reused with `custom:<id>` and appear in full listings, category filters and search. They do not modify the built-in 30-style menu and are excluded from Git and source share packages.

## Visual style

The following lock applies to the default diary style only. Other catalog styles and analyzed reference profiles have independent recipes. The catalog holds 297 styles + 30 palettes, with 30 curated default menu entries.

- Flat white paper
- Uneven felt-tip outlines and sparse wax-crayon color
- Generous negative space
- No realistic shading, glossy gradients, watermark, or paper texture

Built-in previews all depict the same grandmother-and-child bridge scene for comparison. They are style evidence, not story templates. The 30 palette previews share one medium and vary their color family. Source attributions and licenses are retained with the catalog data; public-facing references are collected in the README footer.

## Validation

`npm run check` checks types, reproducible catalog output, isolated workflow fixtures and historical storyboard schemas. Render commands validate their own storyboard's actual image files before starting. Mechanical checks do not establish glyph correctness, style quality or reference fidelity; those require inspecting the generated masters and rendered video.

See the [asset workflow](skill-package/story-to-handdrawn-video/references/asset-workflow.md) and [reference-style workflow](skill-package/story-to-handdrawn-video/references/reference-style-workflow.md) for operational details.
