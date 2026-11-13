// Finalize a complete set of actual project-generated comparison images.
import {createHash} from 'node:crypto';
import {existsSync, readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root=resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read=path=>JSON.parse(readFileSync(resolve(root,path),'utf8'));
const plan=read('references/standard-preview-prompts.json');
const manifestPath='references/standard-preview-manifest.json';
const previous=existsSync(resolve(root,manifestPath)) ? read(manifestPath).images : [];
const images=plan.items.map(item=>{
  const receiptPath=`.dao/standard-preview-receipts/${item.id}.json`;
  const record=existsSync(resolve(root,receiptPath)) ? read(receiptPath) : previous.find(entry=>entry.id===item.id);
  if(!record) throw new Error(`Missing actual generation record: ${item.id}`);
  if(record.id!==item.id||record.path!==item.path||record.scene_id!==plan.scene_id) throw new Error(`Receipt mismatch: ${item.id}`);
  if(record.generator==='image_gen.imagegen' && record.prompt_sha256!==createHash('sha256').update(item.prompt).digest('hex')) throw new Error(`Generation prompt changed: ${item.id}`);
  const bytes=readFileSync(resolve(root,item.path));
  const sha256=createHash('sha256').update(bytes).digest('hex');
  if(sha256!==record.sha256) throw new Error(`Changed image: ${item.id}`);
  return record;
});
if(images.length!==327||new Set(images.map(x=>x.id)).size!==327||new Set(images.map(x=>x.sha256)).size!==327) throw new Error('Expected 327 distinct actual comparison images');
const output={version:1,scene_id:plan.scene_id,scene:plan.scene,
  description:'One shared grandmother-and-child bridge scene; existing project comparisons plus newly generated project artwork. No external library sample images.',
  image_count:images.length,images};
writeFileSync(resolve(root,manifestPath),JSON.stringify(output,null,2)+'\n');
console.log(`Finalized ${images.length} project standard previews`);
