import assert from 'node:assert/strict';
import {execFileSync, spawnSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {copyFileSync, cpSync, existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, symlinkSync, writeFileSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {after, test} from 'node:test';
import {loadStyleLibrary, resolveStyle, resolvePalette} from '../scripts/handdrawn-style-library.mjs';
import {imageHash, prepareReferenceAnalysis, readReferenceProfile, saveReferenceStyle} from '../scripts/reference-style.mjs';
import {dimensionsFor, processPage} from '../scripts/page-assets.mjs';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const sandbox = mkdtempSync(resolve(tmpdir(), 'story-video-regression-'));
const originalHome = process.env.STORY_VIDEO_STYLE_HOME;
process.env.STORY_VIDEO_STYLE_HOME = resolve(sandbox, 'styles');
after(() => {
  if (originalHome === undefined) delete process.env.STORY_VIDEO_STYLE_HOME;
  else process.env.STORY_VIDEO_STYLE_HOME = originalHome;
  rmSync(sandbox, {recursive: true, force: true});
});
const json = (path) => JSON.parse(readFileSync(path, 'utf8'));
const writeJson = (path, data) => writeFileSync(path, `${JSON.stringify(data, null, 2)}\n`);
const cli = (root, script, args = []) => execFileSync(process.execPath, [resolve(root, 'scripts', script), ...args], {cwd: root, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe']});
const workspace = (name) => {
  const root = resolve(sandbox, name);
  mkdirSync(resolve(root, 'public/assets'), {recursive: true});
  cpSync(resolve(repo, 'scripts'), resolve(root, 'scripts'), {recursive: true});
  symlinkSync(resolve(repo, 'references'), resolve(root, 'references'), 'dir');
  copyFileSync(resolve(repo, 'package.json'), resolve(root, 'package.json'));
  return root;
};
const makePage = (path, size = '512x768') => execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', `color=c=white:s=${size}`, '-vf', 'drawbox=x=40:y=30:w=200:h=35:color=black:t=fill,drawbox=x=75:y=330:w=220:h=300:color=blue:t=fill', '-frames:v', '1', '-y', path], {stdio: 'pipe'});
const pathsFor = (name) => {
  const dir = resolve(sandbox, name);mkdirSync(dir, {recursive: true});
  return Object.fromEntries(['master', 'text', 'color', 'bw'].map((key) => [`${key}Path`, resolve(dir, `${key}.png`)]));
};
const profileFor = (image) => ({version: 1, name_zh: '回归淡彩', category: 'watercolor',
  traits: {line: '细黑线', shape: '圆润形状', palette: '蓝色与白色', texture: '轻水彩颗粒', composition: '留白与小主体', lettering: '图片中没有字；配套细笔手写字'},
  confidence: 'medium', best_for: ['日常故事'], avoid: ['厚重油画'], reference_images: [{path: image, sha256: imageHash(image)}]});

// These fixtures test mechanics and contracts, not aesthetic quality or OCR.
test('327 typed assets, exactly 30 defaults, and all legacy selectors/recipes survive', () => {
  const {catalog} = loadStyleLibrary(repo);
  assert.equal(catalog.builtin_style_count, 297);
  assert.equal(catalog.palettes.length, 30);
  assert.equal(new Set(catalog.featured_styles).size, 30);
  for (const legacy of json(resolve(repo, 'references/legacy-handdrawn-styles.json')).styles) {
    for (const selector of [legacy.id, String(legacy.order), legacy.name_zh, ...(legacy.aliases || [])]) {
      assert.equal(resolveStyle(repo, selector).id, legacy.id);
    }
    const current = catalog.styles.find((s) => s.id === legacy.id);
    for (const [key, value] of Object.entries(legacy)) {
      if (key !== 'example_image') assert.deepEqual(current[key], value);
    }
  }
  for (const style of catalog.styles) assert.ok(resolveStyle(repo, style.id).prompt.length > 50);
  assert.equal(resolveStyle(repo, 'HS-277').id, 'hs-277');
  assert.equal(resolveStyle(repo, '21').id, 'hs-001');
  assert.deepEqual(resolveStyle(repo, 'hs-021').references, []);
  assert.throws(() => resolveStyle(repo, '可爱萌系插画风'), /Ambiguous/);
  assert.throws(() => resolveStyle(repo, 'missing'), /Unknown/);
  assert.equal(resolvePalette(repo, 'C-01').id, 'C-01');
});

test('all 327 assets have distinct project bridge previews with verifiable generation records', () => {
  const {catalog} = loadStyleLibrary(repo);
  const assets = [...catalog.styles, ...catalog.palettes];
  assert.equal(new Set(assets.map((asset) => asset.example_image)).size, 327);
  for (const asset of assets) assert.ok(existsSync(resolve(repo, asset.example_image)), asset.id);
  const manifest = json(resolve(repo, 'references/standard-preview-manifest.json'));
  const plan = json(resolve(repo, 'references/standard-preview-prompts.json'));
  assert.equal(manifest.images.length, 327);
  assert.equal(manifest.scene_id, plan.scene_id);
  assert.equal(new Set(manifest.images.map((entry) => entry.sha256)).size, 327);
  assert.equal(manifest.images.filter((entry) => entry.generator === 'project-original').length, 20);
  assert.equal(manifest.images.filter((entry) => entry.generator === 'image_gen.imagegen').length, 307);
  for (const entry of manifest.images) {
    const asset = assets.find((item) => item.id === entry.id);
    const prompt = plan.items.find((item) => item.id === entry.id);
    assert.ok(asset && prompt, entry.id);
    assert.equal(asset.example_image, entry.path);
    assert.equal(entry.path, prompt.path);
    if (entry.generator === 'image_gen.imagegen') {
      assert.equal(entry.prompt_sha256, createHash('sha256').update(prompt.prompt).digest('hex'), entry.id);
    }
    assert.match(entry.path, /^references\/style-examples\/standard\/[a-z0-9-]+\.png$/);
    assert.equal(entry.scene_id, manifest.scene_id);
    assert.equal(asset.example_origin.kind, 'project-standard');
    assert.equal(imageHash(resolve(repo, entry.path)), entry.sha256);
    const bytes = readFileSync(resolve(repo, entry.path));
    assert.deepEqual(bytes.subarray(0, 8), Buffer.from([137,80,78,71,13,10,26,10]));
    assert.equal(bytes.readUInt32BE(16), entry.width);
    assert.equal(bytes.readUInt32BE(20), entry.height);
    assert.equal(entry.width, entry.height);
    assert.ok(entry.width >= 512);
  }
  const gallery = readFileSync(resolve(repo, 'references/style-library.html'), 'utf8');
  assert.ok(!gallery.includes('style-examples/yang0/'));
  assert.ok(!gallery.includes('https://github.com/yang0/'));
});

test('menus expose 30 defaults, 297 styles, 327 assets, and type/category/search filters', () => {
  const menu = (...args) => JSON.parse(cli(repo, 'list-handdrawn-styles.mjs', [...args, '--json']));
  assert.equal(menu().items.length, 30);
  assert.equal(menu('--all').items.length, 297);
  assert.equal(menu('--type', 'all').items.length, 327);
  assert.equal(menu('--type', 'palette').items.length, 30);
  const selected = menu('--category', 'watercolor').items;
  assert.ok(selected.length > 5);
  assert.ok(selected.every((s) => s.category === 'watercolor'));
  assert.equal(menu('--query', 'HS-277').items[0].id, 'hs-277');
});

test('default generated lettering, explicit font fallback, intact page flip, and palette cache isolation', () => {
  const root = workspace('plans');
  const plan = (...args) => {
    cli(root, 'story-to-video.mjs', ['--text', '我走到窗边，看见了一只小鸟。', ...args]);
    return {board: json(resolve(root, 'storyboard.generated.json')), jobs: json(resolve(root, 'codex-image-jobs.json'))};
  };
  const defaults = plan();
  assert.equal(defaults.jobs.text_mode, 'image2');
  assert.equal(defaults.jobs.style_id, 'colored-pencil-diary');
  assert.match(defaults.board.scenes[0].assets.text_image, /_text.png$/);
  assert.equal(defaults.jobs.jobs[1].size, '1024x1536');
  assert.ok(defaults.jobs.jobs[1].prompt.includes(defaults.jobs.jobs[1].expected_caption));
  const font = plan('--text-mode', 'font');
  assert.equal(font.board.scenes[0].assets.text_image, null);
  assert.notEqual(font.jobs.asset_set, defaults.jobs.asset_set);
  const themed = plan('--palette', 'C-01');
  assert.notEqual(themed.jobs.asset_set, defaults.jobs.asset_set);
  assert.match(themed.jobs.jobs[1].prompt, /Explicit palette override/);
  const other = plan('--style', 'HS-003');
  assert.equal(other.jobs.style_references.length, 0);
  assert.notEqual(other.jobs.asset_set, defaults.jobs.asset_set);
  const flipped = plan('--transition', 'page-flip');
  assert.deepEqual(flipped.board.scenes[0].layers, ['color']);
  assert.match(flipped.board.scenes[0].assets.color, /_master.png$/);
  assert.equal(flipped.board.scenes[0].text, '');
  assert.throws(() => plan('--transition', 'page-flip', '--text-mode', 'font'), /Page-flip requires/);
});

test('reference analysis request, profile validation, private save and reuse after source removal', () => {
  const root = workspace('reference');
  const image = resolve(root, 'input.png');makePage(image);
  const prepared = prepareReferenceAnalysis(root, [image]);
  assert.equal(prepared.request.reference_images[0].sha256, imageHash(image));
  assert.equal(existsSync(prepared.request.output_profile), false);
  const profilePath = prepared.request.output_profile;
  writeJson(profilePath, profileFor(image));
  assert.equal(readReferenceProfile(profilePath).traits.line, '细黑线');
  cli(root, 'story-to-video.mjs', ['--text', '小鸟飞来了。', '--style-profile', profilePath]);
  const jobs = json(resolve(root, 'codex-image-jobs.json'));
  assert.deepEqual(jobs.style_references, [image]);
  assert.match(jobs.jobs[1].prompt, /图片中没有字/);
  assert.match(jobs.jobs[1].prompt, /Avoid: 厚重油画/);
  assert.throws(() => saveReferenceStyle(root, profilePath, '../escape'), /style id/);
  const saved = saveReferenceStyle(root, profilePath, 'watercolor-test');
  assert.throws(() => saveReferenceStyle(root, profilePath, 'watercolor-test'), /already exists/);
  rmSync(image);
  const selected = resolveStyle(root, saved.id);
  assert.ok(existsSync(selected.references[0].absolute_path));
  assert.equal(selected.category, 'watercolor');
  cli(root, 'story-to-video.mjs', ['--text', '小鸟飞来了。', '--style', saved.id]);
  assert.equal(json(resolve(root, 'codex-image-jobs.json')).style_id, saved.id);
  assert.ok(!readFileSync(resolve(saved.path, 'profile.json'), 'utf8').includes(sandbox));
  // Damaged input must not silently resolve as the default diary style.
  writeFileSync(selected.references[0].absolute_path, 'changed');
  assert.throws(() => resolveStyle(root, saved.id), /changed; re-analyze/);
  rmSync(saved.path, {recursive: true});
});

test('shared crop derives aligned plates without modifying the master, including scaled generated masters', () => {
  const generated = pathsFor('generated-plates');makePage(generated.masterPath);
  const hash = imageHash(generated.masterPath);
  processPage({input: generated.masterPath, ...generated, generatedTextMode: 'image2'});
  assert.equal(imageHash(generated.masterPath), hash);
  assert.deepEqual(dimensionsFor(generated.textPath), {width: 1536, height: 512});
  assert.deepEqual(dimensionsFor(generated.colorPath), {width: 1024, height: 1024});
  assert.deepEqual(dimensionsFor(generated.bwPath), dimensionsFor(generated.colorPath));
  const uploaded = pathsFor('uploaded-plates');
  processPage({input: generated.masterPath, ...uploaded, layout: 'composite'});
  assert.equal(imageHash(uploaded.textPath), imageHash(generated.textPath));
  assert.equal(imageHash(uploaded.colorPath), imageHash(generated.colorPath));
  assert.equal(imageHash(uploaded.bwPath), imageHash(generated.bwPath));
  const flipped = pathsFor('flipped-plates');
  processPage({input: generated.masterPath, ...flipped, transition: 'page-flip', generatedTextMode: 'image2'});
  assert.equal(imageHash(flipped.masterPath), hash);
  for (const key of ['textPath', 'colorPath', 'bwPath']) assert.equal(existsSync(flipped[key]), false);
  const bad = pathsFor('bad-plates');makePage(bad.masterPath, '512x512');
  assert.throws(() => processPage({input: bad.masterPath, ...bad, generatedTextMode: 'image2'}), /must be 2:3/);
});

test('generated art above the requested one-third boundary survives; intact flips need no crop gutter', () => {
  const early = pathsFor('early-art');
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=white:s=512x768', '-vf', 'drawbox=x=40:y=30:w=200:h=35:color=black:t=fill,drawbox=x=75:y=205:w=220:h=300:color=blue:t=fill', '-frames:v', '1', '-y', early.masterPath]);
  const geometry = processPage({input: early.masterPath, ...early, generatedTextMode: 'image2'});
  assert.ok(geometry.splitY > 65 && geometry.splitY < 205, `split at ${geometry.splitY} must stay in the actual white gutter`);
  const blank = pathsFor('blank');
  execFileSync('ffmpeg', ['-hide_banner', '-loglevel', 'error', '-f', 'lavfi', '-i', 'color=c=white:s=512x768', '-frames:v', '1', '-y', blank.masterPath]);
  assert.throws(() => processPage({input: blank.masterPath, ...blank, generatedTextMode: 'image2'}), /No clear caption/);
  const full = pathsFor('full-page-without-gutter');
  const hash = imageHash(blank.masterPath);
  processPage({input: blank.masterPath, ...full, generatedTextMode: 'image2', transition: 'page-flip'});
  assert.equal(imageHash(full.masterPath), hash);
  assert.equal(existsSync(full.textPath), false);
});

for (const transition of ['cut', 'page-flip']) {
  test(`executed generated plan → fixture masters → import → strict validation (${transition})`, () => {
    const root = workspace(`import-${transition}`);
    cli(root, 'story-to-video.mjs', ['--text', '小鸟飞来了。', '--transition', transition]);
    const jobs = json(resolve(root, 'codex-image-jobs.json'));
    for (const job of jobs.jobs) makePage(job.output_master, job.role === 'reference' ? '512x512' : '512x768');
    const masterHash = imageHash(jobs.jobs[1].output_master);
    cli(root, 'import-codex-images.mjs', ['--apply']);
    cli(root, 'validate-storyboard.mjs', ['storyboard.json']);
    assert.equal(imageHash(jobs.jobs[1].output_master), masterHash);
    const scene = json(resolve(root, 'storyboard.json')).scenes[0];
    if (transition === 'cut') {
      assert.deepEqual(scene.layers, ['text', 'bw_full', 'color']);
      scene.assets.text_image = null;
      const board = json(resolve(root, 'storyboard.json'));board.scenes[0] = scene;writeJson(resolve(root, 'storyboard.json'), board);
      assert.throws(() => cli(root, 'validate-storyboard.mjs', ['storyboard.json']), /no automatic font fallback/);
    } else {
      assert.equal(scene.shot, 'full_generated_page');
      assert.deepEqual(scene.layers, ['color']);
    }
  });
  test(`uploaded import stays compatible (${transition})`, () => {
    const root = workspace(`uploaded-${transition}`);
    const input = resolve(root, 'page.png');makePage(input);
    cli(root, 'import-uploaded-pages.mjs', ['--image', input, '--transition', transition, '--layout', 'composite', '--split-y', '01:256']);
    cli(root, 'validate-storyboard.mjs', ['storyboard.uploaded.json']);
    const scene = json(resolve(root, 'storyboard.uploaded.json')).scenes[0];
    assert.deepEqual(scene.layers, transition === 'cut' ? ['text', 'bw_full', 'color'] : ['color']);
    if (transition === 'page-flip') assert.equal(imageHash(resolve(root, 'public', scene.assets.color)), imageHash(input));
  });
}

test('portable Python wrapper forwards catalog options and rejects mismatched references', () => {
  const menu = execFileSync('python3', [resolve(repo, 'skill-package/story-to-handdrawn-video/scripts/run_story_video.py'), '--list-styles', '--asset-type', 'palette', '--json'], {cwd: repo, encoding: 'utf8'});
  assert.equal(JSON.parse(menu).items.length, 30);
  const root = workspace('wrapper');
  const image = resolve(root, 'one.png');makePage(image);
  const other = resolve(root, 'two.png');copyFileSync(image, other);
  const profile = resolve(root, 'profile.json');writeJson(profile, profileFor(image));
  const result = spawnSync('python3', [resolve(root, 'scripts/run_story_video.py'), '--text', '小鸟来了。', '--style-reference', other, '--style-profile', profile], {cwd: root, encoding: 'utf8'});
  assert.notEqual(result.status, 0);
  assert.match(result.stderr, /differ from the analyzed profile/);
});
