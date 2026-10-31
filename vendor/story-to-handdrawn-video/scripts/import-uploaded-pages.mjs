import {createHash} from 'node:crypto';
import {
  existsSync,
  mkdirSync,
  readFileSync,
  writeFileSync,
} from 'node:fs';
import {dirname, extname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {processPage} from './page-assets.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const parseArgs = (tokens) => {
  const parsed = {images: [], splitYs: []};
  for (let index = 0; index < tokens.length; index += 1) {
    const token = tokens[index];
    if (token === '--image') {
      const value = tokens[index + 1];
      if (!value) throw new Error('--image requires a path');
      parsed.images.push(value);
      index += 1;
      continue;
    }
    if (token === '--split-y') {
      const value = tokens[index + 1];
      if (!value) throw new Error('--split-y requires SCENE:PIXELS');
      parsed.splitYs.push(value);
      index += 1;
      continue;
    }
    if (!token.startsWith('--')) continue;
    const key = token.slice(2);
    const next = tokens[index + 1];
    if (next && !next.startsWith('--')) {
      parsed[key] = next;
      index += 1;
    } else {
      parsed[key] = true;
    }
  }
  return parsed;
};

const args = parseArgs(process.argv.slice(2));
if (args.images.length === 0) {
  throw new Error(
    'Usage: npm run import:uploaded -- --image /path/01.jpg --image /path/02.jpg [--transition page-flip]',
  );
}

const title = String(args.title || '上传图片手绘动画');
// Page turning is opt-in. Uploaded stories use direct cuts unless the user
// explicitly requests a paper-page transition.
const transition = String(args.transition || 'cut');
const transitionSec = Number(args['transition-sec'] || 0.7);
const pageDuration = Number(args['page-duration'] || 4.4);
const layout = String(args.layout || 'auto');
const paperFlip = transition === 'page-flip';
const splitOverrides = new Map(
  args.splitYs.map((value) => {
    const match = /^(\d+):(\d+)$/.exec(String(value));
    if (!match) throw new Error(`Invalid --split-y value: ${value}`);
    return [match[1].padStart(2, '0'), Number(match[2])];
  }),
);

if (!['cut', 'page-flip'].includes(transition)) {
  throw new Error('--transition must be cut or page-flip');
}
if (!['auto', 'composite', 'full'].includes(layout)) {
  throw new Error('--layout must be auto, composite, or full');
}
if (!Number.isFinite(transitionSec) || transitionSec <= 0 || transitionSec > 2) {
  throw new Error('--transition-sec must be greater than 0 and at most 2');
}
if (!Number.isFinite(pageDuration) || pageDuration < 2 || pageDuration > 15) {
  throw new Error('--page-duration must be between 2 and 15 seconds');
}

const resolvedInputs = args.images.map((value) => resolve(root, String(value)));
for (const input of resolvedInputs) {
  if (!existsSync(input)) throw new Error(`Uploaded image does not exist: ${input}`);
}

const seenHashes = new Set();
const inputs = [];
for (const input of resolvedInputs) {
  const hash = createHash('sha256').update(readFileSync(input)).digest('hex');
  if (seenHashes.has(hash)) {
    console.log(`Skipped exact duplicate: ${input}`);
    continue;
  }
  seenHashes.add(hash);
  inputs.push({path: input, hash});
}

if (inputs.length === 0) throw new Error('No unique uploaded images remain');

const safeTitle =
  title
    .normalize('NFKC')
    .replace(/[^\p{Letter}\p{Number}]+/gu, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 32) || 'uploaded';
const batchHash = createHash('sha256')
  .update(
    [
      'shared-page-assets-v1',
      layout,
      transition,
      paperFlip ? String(transitionSec) : 'no-transition-overlap',
      String(pageDuration),
      JSON.stringify([...splitOverrides]),
      ...inputs.map((item) => item.hash),
    ].join('\n'),
  )
  .digest('hex')
  .slice(0, 8);
const assetSet = `${safeTitle}-${batchHash}`;
const generatedRoot = `generated/uploads/${assetSet}`;
const outputDir = resolve(root, 'public/assets', generatedRoot);
mkdirSync(outputDir, {recursive: true});

const scenes = [];
const manifestPages = [];

for (let index = 0; index < inputs.length; index += 1) {
  const input = inputs[index];
  const id = String(index + 1).padStart(2, '0');
  const extension = extname(input.path).toLowerCase() || '.jpg';
  const masterName = `${id}_master${extension}`;
  const textName = `${id}_text.png`;
  const colorName = `${id}_color.png`;
  const bwName = `${id}_bw.png`;
  const masterPath = resolve(outputDir, masterName);
  const textPath = resolve(outputDir, textName);
  const colorPath = resolve(outputDir, colorName);
  const bwPath = resolve(outputDir, bwName);

  const {width, height, hasCaption, splitY, captionY, captionH} = processPage({
    input: input.path, masterPath, textPath, colorPath, bwPath,
    transition, layout, splitOverride: splitOverrides.get(id),
  });

  const asset = (name) => `assets/${generatedRoot}/${name}`;
  scenes.push({
    id,
    duration_sec: pageDuration,
    text: '',
    visual: `上传图片 ${id}`,
    shot: paperFlip ? 'full_uploaded_page' : 'safe_uncropped_uploaded_page',
    // Page curls use the untouched vertical master. This preserves every
    // handwritten line and avoids auto-crop errors on long captions.
    layers: paperFlip
      ? ['color']
      : hasCaption
        ? ['text', 'bw_full', 'color']
        : ['bw_full', 'color'],
    color_hint: null,
    detail_hint: null,
    caption_box: null,
    assets: {
      text_image: !paperFlip && hasCaption ? asset(textName) : null,
      bw: paperFlip ? null : asset(bwName),
      detail: null,
      color: asset(paperFlip ? masterName : colorName),
    },
  });
  manifestPages.push({
    id,
    source: input.path,
    master: masterPath,
    width,
    height,
    has_caption: hasCaption,
    detected_split_y: splitY,
    caption_crop: hasCaption
      ? {y: captionY, height: captionH}
      : null,
  });
}

const storyboard = {
  project: {
    title,
    mode: 'speed',
    images_per_scene: 1,
    derive_bw: 'local',
    enable_detail: false,
    gen_size: 1024,
    export_size: [1080, 1440],
    ratio: '3:4',
    width: 1080,
    height: 1440,
    fps: 30,
    transition,
    transition_sec: paperFlip ? transitionSec : undefined,
    style_lock: 'preserve uploaded hand-drawn diary-comic artwork',
    character_lock: 'preserve uploaded characters and composition exactly',
    audio: {
      voiceover: 'post',
      bgm: 'optional_bed_only',
      bgm_follows_text: false,
    },
  },
  scenes,
};

const storyboardPath = resolve(root, 'storyboard.uploaded.json');
const manifestPath = resolve(root, 'uploaded-pages.json');
writeFileSync(storyboardPath, `${JSON.stringify(storyboard, null, 2)}\n`);
writeFileSync(
  manifestPath,
  `${JSON.stringify(
    {
      version: 1,
      asset_set: assetSet,
      storyboard: storyboardPath,
      transition,
      pages: manifestPages,
    },
    null,
    2,
  )}\n`,
);

const transitionOverlap =
  transition === 'page-flip' ? transitionSec * Math.max(0, scenes.length - 1) : 0;
const duration = scenes.length * pageDuration - transitionOverlap;
console.log(
  `Prepared ${scenes.length} uploaded scenes (${duration.toFixed(1)}s, ${transition}) → ${storyboardPath}`,
);
console.log(`Assets → ${outputDir}`);
