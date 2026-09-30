// Register actual ImageGen outputs. This script never generates or redraws art.
import {createHash} from 'node:crypto';
import {copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync} from 'node:fs';
import {basename, dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const receipts = resolve(root, '.dao/standard-preview-receipts');
const [id, source] = process.argv.slice(2);
const plan = JSON.parse(readFileSync(resolve(root, 'references/standard-preview-prompts.json'), 'utf8'));
const item = plan.items.find((entry) => entry.id === id);
if (!item || !source) throw new Error('Usage: node scripts/register-standard-preview.mjs <catalog-id> <actual-imagegen-output.png>');
const bytes = readFileSync(source);
if (!bytes.subarray(0, 8).equals(Buffer.from([137,80,78,71,13,10,26,10]))) throw new Error('Expected a real PNG output');
const width = bytes.readUInt32BE(16), height = bytes.readUInt32BE(20);
if (width !== height || width < 512) throw new Error(`Expected a square master, got ${width}x${height}`);
const target = resolve(root, item.path);
if (existsSync(target)) throw new Error(`Already registered: ${id}`);
mkdirSync(dirname(target), {recursive: true});
copyFileSync(source, target);
const receipt = {id, path:item.path, name:item.name, type:item.type,
  scene_id:plan.scene_id, generator:plan.generator, model:plan.model,
  generation_mode:'text-only', source_output:basename(source),
  prompt_sha256:createHash('sha256').update(item.prompt).digest('hex'),
  sha256:createHash('sha256').update(bytes).digest('hex'), width, height,
  visual_review:'pending'};
mkdirSync(receipts, {recursive:true});
writeFileSync(resolve(receipts, `${id}.json`), JSON.stringify(receipt,null,2)+'\n');
console.log(JSON.stringify({id,path:item.path,width,height}));
