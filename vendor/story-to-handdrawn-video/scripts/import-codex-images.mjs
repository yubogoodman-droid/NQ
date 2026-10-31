import {execFileSync} from 'node:child_process';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, relative, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {processPage} from './page-assets.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const args = {};
for (let index = 2; index < process.argv.length; index += 1) {
  const token = process.argv[index];
  if (!token.startsWith('--')) continue;
  const next = process.argv[index + 1];
  args[token.slice(2)] = next && !next.startsWith('--') ? process.argv[++index] : true;
}
if (args.render && !args.apply) throw new Error('--render requires --apply');
const manifestPath = resolve(root, String(args.manifest || 'codex-image-jobs.json'));
const manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
if (manifest.generator !== 'codex-image2' || !Array.isArray(manifest.jobs)) throw new Error('Unsupported Codex image manifest');
const missing = manifest.jobs.map((job) => job.output_master).filter((master) => !existsSync(master));
if (missing.length) throw new Error(`Codex masters are incomplete (${missing.length} missing):\n${missing.join('\n')}`);
const storyboard = JSON.parse(readFileSync(resolve(manifest.storyboard), 'utf8'));
const transition = manifest.transition || storyboard.project.transition || 'cut';
const textMode = manifest.text_mode || 'image2';
if (!['font', 'image2'].includes(textMode)) throw new Error('Unknown manifest text_mode');
if (transition === 'page-flip' && textMode === 'font') throw new Error('Page-flip requires an intact caption + illustration master (--text-mode image2)');
for (const job of manifest.jobs.filter((item) => item.role !== 'reference')) {
  const scene = storyboard.scenes.find((item) => item.id === job.id);
  if (!scene) throw new Error(`Missing scene ${job.id} in storyboard`);
  const masterPath = resolve(job.output_master);
  const assetDir = dirname(masterPath);
  const textPath = resolve(assetDir, `${job.id}_text.png`);
  const colorPath = resolve(assetDir, `${job.id}_color.png`);
  const bwPath = resolve(assetDir, `${job.id}_bw.png`);
  processPage({input: masterPath, masterPath, textPath, colorPath, bwPath, transition, generatedTextMode: textMode});
  const asset = (path) => relative(resolve(root, 'public'), path).split('\\').join('/');
  if (transition === 'page-flip') {
    scene.text = '';
    scene.shot = 'full_generated_page';
    scene.layers = ['color'];
    scene.assets = {text_image: null, bw: null, detail: null, color: asset(masterPath)};
  } else {
    scene.assets = {text_image: textMode === 'image2' ? asset(textPath) : null,
      bw: asset(bwPath), detail: null, color: asset(colorPath)};
  }
  console.log(`Imported scene ${job.id} → ${assetDir}`);
}
if (args.apply) {
  writeFileSync(resolve(root, 'storyboard.json'), `${JSON.stringify(storyboard, null, 2)}\n`);
  console.log(`Activated storyboard → ${resolve(root, 'storyboard.json')}`);
}
if (args.render) execFileSync('npm', ['run', 'render'], {cwd: root, stdio: 'inherit'});
