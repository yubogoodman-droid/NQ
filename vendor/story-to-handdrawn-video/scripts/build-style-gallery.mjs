import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const catalog = JSON.parse(readFileSync(resolve(root, 'references/handdrawn-style-library.json'), 'utf8'));
// Only public built-ins are embedded. User reference photos never enter this artifact.
const data = {featured: catalog.featured_styles, categories: catalog.categories,
  items: [...catalog.styles.map((s) => ({id:s.id,type:'style',category:s.category,name:s.name_zh.replace(/（HS-\d+）$/, ''),
    english:s.name_en,summary:s.summary,keywords:s.best_for.join(' '),image:s.example_image?.replace(/^references\//,'') || null})),
    ...catalog.palettes.map((p) => ({id:p.id,type:'palette',category:p.category,categoryName:p.category_zh,
      name:p.name_zh,english:p.name_en,image:p.example_image.replace(/^references\//,''),summary:'以'+p.name_zh+'为主要点缀色，保留所选画材的质感、明暗层次与清晰文字。',keywords:''}))]};
const serialized = JSON.stringify(data).replace(/</g, '\\u003c');
const html = `<!doctype html>
<html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>手绘主题资产库 · 327</title>
<style>
:root{color-scheme:light;font-family:system-ui,-apple-system,"PingFang SC",sans-serif;color:#262a28;background:#f5f5ef}*{box-sizing:border-box}body{margin:0}main{max-width:1240px;margin:auto;padding:48px 24px}header{border-bottom:2px solid #263d34;padding-bottom:30px}.eyebrow{font-size:12px;letter-spacing:.16em;color:#456354}h1{font-size:clamp(30px,5vw,54px);margin:12px 0 16px;font-weight:650}p{line-height:1.7;color:#56625b}.counts{display:flex;gap:12px;flex-wrap:wrap}.counts span{padding:8px 12px;background:#e6ece4;border-radius:6px;font-size:13px}.controls{display:flex;gap:10px;flex-wrap:wrap;margin:28px 0 16px}input,select,button{font:inherit;font-size:14px;padding:12px;border:1px solid #bac8bc;border-radius:8px;background:white;color:inherit}input{flex:1;min-width:200px}button{cursor:pointer}button[aria-pressed=true]{background:#263d34;color:white}button:focus-visible,input:focus-visible,select:focus-visible{outline:3px solid #cba651;outline-offset:2px}.status{font-size:13px;min-height:24px}#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:18px}.card{background:white;border:1px solid #d9e0d6;border-radius:12px;overflow:hidden;display:flex;flex-direction:column}.visual{height:205px;overflow:hidden;display:grid;place-items:center;background:#fff;border-bottom:1px solid #e7ebe2}.visual a{display:block;width:100%;height:205px}.visual img{display:block;width:100%;height:205px;object-fit:contain}.empty{padding:25px;color:#748175;font-size:13px;text-align:center;line-height:1.8}.body{padding:20px;display:flex;flex:1;flex-direction:column}.badge{font-size:11px;text-transform:uppercase;letter-spacing:.05em;color:#4d6b57}h2{font-size:17px;line-height:1.5;margin:10px 0 6px}.english{font-size:12px;color:#7b857d}.description{font-size:13px;line-height:1.75;flex:1}.use{width:100%;text-align:left;font-family:ui-monospace,monospace;font-size:12px;background:#f4f7f1}.footer{font-size:12px;margin-top:30px}a{color:#31563e}#copied{font-size:13px;color:#31563e;min-height:22px}
.preview{border:1px solid #bac8bc;border-radius:12px;padding:20px;max-width:92vw;max-height:92vh;background:#fff}.preview::backdrop{background:#14281bd9}.preview header{display:flex;align-items:center;gap:24px;justify-content:space-between;border:0;padding:0 0 12px}.preview h2{margin:0}.preview img{display:block;width:min(760px,78vw);height:auto;max-height:72vh;object-fit:contain}.preview p{font-size:12px;margin:12px 0 0}
</style></head><body><main>
<header><div class="eyebrow">STORY TO HAND-DRAWN VIDEO / ASSET LIBRARY</div><h1>选一种画风，讲你的故事。</h1>
<p>同一座桥，同一对老人和小孩。用统一场景比较线条、画材与配色，选出适合你故事的表达。</p>
<div class="counts"><span>297 种画风配方</span><span>30 种主题配色</span><span>默认精选 30</span><span>327 张标准配图</span></div></header>
<div class="controls"><button id="featured" aria-pressed="true">常用精选</button><button id="all" aria-pressed="false">全部资产</button>
<select id="type" aria-label="资产类型"><option value="all">全部类型</option><option value="style">画风</option><option value="palette">主题配色</option></select>
<select id="category" aria-label="资产分类"><option value="">全部分类</option></select>
<input id="search" type="search" aria-label="搜索资产" placeholder="搜索名称、编号或题材，例如 水彩 / HS-128"></div>
<p class="status" id="status" aria-live="polite"></p><div id="copied" role="status"></div><section id="grid" aria-label="资产列表"></section>
<p class="footer">全部配图统一采用“老人和小孩过桥”场景，展示不同画材与配色的表现。精选按常见用途选编；个人收藏请用 Skill 的风格菜单查看。<br>
<a href="../README.md">项目使用说明</a> · <a href="../README.md#参考与许可">参考与许可</a></p>
</main><dialog class="preview" id="preview" aria-labelledby="preview-title"><header><h2 id="preview-title"></h2><form method="dialog"><button aria-label="关闭示意图">关闭 ×</button></form></header><img id="preview-image" alt=""><p>统一过桥场景 · 本项目标准配图</p></dialog><script id="catalog" type="application/json">${serialized}</script><script>
const data=JSON.parse(document.getElementById('catalog').textContent);const byId=id=>document.getElementById(id);let featured=true;
const categories=new Map(data.categories.map(c=>[c.id,c.name_zh]));for(const item of data.items)if(item.categoryName)categories.set(item.category,item.categoryName);
for(const [id,name] of categories){const option=document.createElement('option');option.value=id;option.textContent=name;byId('category').append(option)}
function element(tag,text,className){const node=document.createElement(tag);if(text)node.textContent=text;if(className)node.className=className;return node}
function render(){const query=byId('search').value.trim().toLowerCase(),type=byId('type').value,category=byId('category').value;
let items=featured?data.featured.map(id=>data.items.find(i=>i.id===id)):data.items;
items=items.filter(i=>(type==='all'||i.type===type)&&(!category||i.category===category)&&(!query||[i.name,i.english,i.id,i.summary,i.keywords].join(' ').toLowerCase().includes(query)));
byId('featured').setAttribute('aria-pressed',String(featured));byId('all').setAttribute('aria-pressed',String(!featured));byId('status').textContent=(featured?'常用精选':'完整目录')+' · '+items.length+' / 327 项';byId('grid').replaceChildren();
for(const item of items){const card=element('article',null,'card'),visual=element('div',null,'visual');if(item.image){const img=element('img');img.src=item.image;img.alt=item.name+' 示例';img.loading='lazy';const link=element('a');link.href=item.image;link.setAttribute('aria-label','放大查看 '+item.name);link.onclick=event=>{event.preventDefault();byId('preview-title').textContent=item.name;byId('preview-image').src=item.image;byId('preview-image').alt=item.name+' 原始示意图';byId('preview').showModal()};link.append(img);visual.append(link)}else{visual.append(element('div',item.type==='palette'?'主题配色配方':'画风配方 · 暂无本地示例图','empty'))}card.append(visual);
const body=element('div',null,'body');body.append(element('div',(categories.get(item.category)||item.category)+' / '+item.id,'badge'),element('h2',item.name),element('div',item.english,'english'),element('p',item.summary,'description'));
const command=(item.type==='style'?'--style ':'--palette ')+item.id,button=element('button',command+'  ·  复制','use');button.onclick=async()=>{try{await navigator.clipboard.writeText(command);byId('copied').textContent='已复制 '+command}catch{byId('copied').textContent='可复制命令：'+command}};body.append(button);card.append(body);byId('grid').append(card)}}
byId('featured').onclick=()=>{featured=true;byId('search').value='';byId('type').value='all';byId('category').value='';render()};byId('all').onclick=()=>{featured=false;render()};for(const id of ['search','type','category'])byId(id).addEventListener('input',()=>{featured=false;render()});render();
</script></body></html>\n`;
const output=resolve(root,'references/style-library.html');
if(process.argv.includes('--check')){if(readFileSync(output,'utf8')!==html)throw new Error('Gallery is stale: run npm run styles:build');}
else writeFileSync(output,html);
console.log('Offline catalog: references/style-library.html');
