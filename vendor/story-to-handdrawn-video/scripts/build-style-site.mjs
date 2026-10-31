import {createHash} from 'node:crypto';
import {spawn} from 'node:child_process';
import {copyFileSync, existsSync, lstatSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync} from 'node:fs';
import {dirname, resolve, relative} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const output = resolve(root, 'site-dist');
const repository = 'https://github.com/gnipbao/story-to-handdrawn-video';
const revision = '57b17c5447d933611617d5661e6c73ba48f06b49';
const originals = `https://raw.githubusercontent.com/gnipbao/story-to-handdrawn-video/${revision}/references/`;
const videoName = 'handdrawn-styles-75-page-flip-45s-bgm.mp4';
const sourceVideo = resolve(root, 'examples/style-showcase', videoName);
const encoderSettings = 'webp:thumb440-q82-preview1120-q88-v1';
const ffmpeg = process.env.FFMPEG_PATH || 'ffmpeg';
const sha256 = (path) => createHash('sha256').update(readFileSync(path)).digest('hex');
const escape = (value) => String(value).replace(/[&<>"']/g, (char) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[char]));
const fail = (message) => {throw new Error(message);};

if (existsSync(output) && lstatSync(output).isSymbolicLink()) fail('site-dist must be a regular directory.');
for (const dir of ['assets/thumbs', 'assets/previews', 'assets/video', 'demo', 'licenses']) mkdirSync(resolve(output, dir), {recursive:true});
const cachePath = resolve(output, '.site-build.json');
const previous = existsSync(cachePath) ? JSON.parse(readFileSync(cachePath, 'utf8')) : {};
const sourceHtml = readFileSync(resolve(root, 'references/style-library.html'), 'utf8');
const catalogPattern = /(<script id="catalog" type="application\/json">)([\s\S]*?)(<\/script>)/;
const catalogMatch = sourceHtml.match(catalogPattern);
if (!catalogMatch) fail('The offline gallery has no public catalog. Run npm run styles:build.');
const data = JSON.parse(catalogMatch[2]);
if (data.items.length !== 327 || data.featured.length !== 30 || new Set(data.items.map((item) => item.id)).size !== 327) fail('Expected 327 unique assets and 30 featured styles.');
const current = {encoder:encoderSettings, originalRevision:revision, assets:{}};

function run(args) {
  return new Promise((resolvePromise, reject) => {
    const process = spawn(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-nostdin', '-y', ...args], {stdio:['ignore','ignore','pipe']});
    let stderr = '';
    process.stderr.on('data', (chunk) => {stderr += chunk;});
    process.once('error', reject);
    process.once('close', (code) => code === 0 ? resolvePromise() : reject(new Error(`FFmpeg failed (${code}): ${stderr}`)));
  });
}

let completed = 0;
const pending = [...data.items];
async function convert() {
  while (pending.length) {
    const item = pending.shift();
    if (!/^style-examples\/standard\/[a-z0-9-]+\.png$/.test(item.image)) fail(`Unexpected public image path for ${item.id}`);
    const source = resolve(root, 'references', item.image);
    const digest = sha256(source);
    const slug = item.id.toLowerCase();
    const thumb = `assets/thumbs/${slug}.webp`;
    const preview = `assets/previews/${slug}.webp`;
    if (previous.encoder !== encoderSettings || previous.assets?.[item.id]?.sourceSha256 !== digest || !existsSync(resolve(output, thumb)) || !existsSync(resolve(output, preview))) {
      await run(['-i', source, '-filter_complex', '[0:v]split=2[t][p];[t]scale=440:440:force_original_aspect_ratio=decrease:flags=lanczos[thumb];[p]scale=1120:1120:force_original_aspect_ratio=decrease:flags=lanczos[preview]',
        '-map', '[thumb]', '-frames:v', '1', '-c:v', 'libwebp', '-quality', '82', '-compression_level', '6', resolve(output, thumb),
        '-map', '[preview]', '-frames:v', '1', '-c:v', 'libwebp', '-quality', '88', '-compression_level', '6', resolve(output, preview)]);
    }
    current.assets[item.id] = {sourceSha256:digest, thumbnail:thumb, preview};
    item.original = originals + item.image;
    item.image = thumb;
    item.preview = preview;
    completed += 1;
    if (completed % 50 === 0 || completed === data.items.length) console.log(`WebP assets: ${completed} / ${data.items.length}`);
  }
}
await Promise.all(Array.from({length:4}, convert));
const poster = resolve(output, 'assets/video/poster.webp');
const sourcePoster = resolve(root, 'examples/style-showcase/poster.png');
await run(['-i', sourcePoster, '-vf', 'scale=720:960:force_original_aspect_ratio=decrease:flags=lanczos', '-frames:v', '1', '-c:v', 'libwebp', '-quality', '88', '-compression_level', '6', poster]);
copyFileSync(sourceVideo, resolve(output, 'assets/video', videoName));
copyFileSync(resolve(root, 'LICENSE'), resolve(output, 'licenses/project-MIT.txt'));
copyFileSync(resolve(root, 'references/vendor/yang0-handraw-style/LICENSE'), resolve(output, 'licenses/style-metadata-MIT.txt'));
copyFileSync(resolve(root, 'references/handdrawn-styles-LICENSE.txt'), resolve(output, 'licenses/original-styles-LICENSE.txt'));
copyFileSync(resolve(root, 'examples/style-showcase/BGM_CREDITS.md'), resolve(output, 'licenses/BGM_CREDITS.md'));
copyFileSync(resolve(root, 'site/vercel.json'), resolve(output, 'vercel.json'));
writeFileSync(resolve(output, '.vercelignore'), '.site-build.json\n.vercel/\n.env*\n.gitignore\n');

const extraCss = `
.site-nav{display:flex;gap:18px;align-items:center;flex-wrap:wrap;font-size:14px;margin-bottom:24px}.showcase{margin:32px 0;padding:28px;border:1px solid #d9e0d6;border-radius:14px;background:#fff;display:grid;grid-template-columns:minmax(0,1fr) minmax(220px,300px);gap:28px;align-items:center}.showcase h2{font-size:28px;margin:0 0 14px}.showcase p{font-size:14px}.showcase video{display:block;width:100%;max-height:460px;aspect-ratio:3/4;background:#fff;border:1px solid #e7ebe2;border-radius:8px}.showcase .music{font-size:11px;line-height:1.7}.preview{max-width:calc(100vw - 24px);padding:16px}.preview header{gap:12px}.preview h2{overflow-wrap:anywhere}.preview img{max-width:100%;width:min(760px,78vw)}.preview-links{display:flex;gap:14px;flex-wrap:wrap}.demo-page{max-width:920px}.demo-page .showcase{grid-template-columns:minmax(0,1fr) minmax(260px,400px)}.footer{overflow-wrap:anywhere}video:focus-visible,a:focus-visible{outline:3px solid #cba651;outline-offset:3px}
@media(max-width:640px){main{padding:28px 16px}.showcase,.demo-page .showcase{padding:20px;grid-template-columns:1fr;gap:18px}.showcase h2{font-size:24px}.showcase video{max-width:340px;max-height:454px;justify-self:center}.controls input{min-width:100%;width:100%}.controls select{max-width:100%}.preview img{width:100%;max-height:66vh}#grid{grid-template-columns:minmax(0,1fr)}}
`;
const credits = `<span>音乐：<a href="https://incompetech.com/music/royalty-free/index.html?isrc=USUAN1400037">Carefree — Kevin MacLeod</a>，<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>；45 秒节选，音量调整与淡入淡出。</span>`;
function showcase(prefix = '', standalone = false) {
  return `<section class="showcase" aria-labelledby="showcase-title"><div><div class="eyebrow">75 STYLES / 45 SECONDS</div><h2 id="showcase-title">翻过 75 种手绘风格。</h2><p>同一个故事场景，跨越 11 类画材与表现方式。播放这段 45 秒翻书动画，看看线条、纸张和颜色如何改变故事的气质。</p><p>${standalone ? '<a href="../">浏览全部 327 项资产 →</a>' : '<a href="demo/">打开独立视频页 →</a>'} · <a href="${prefix}assets/video/${videoName}" download>下载视频</a></p><p class="music">${credits}<br><a href="${prefix}licenses/BGM_CREDITS.md">完整音乐署名</a></p></div><video controls playsinline preload="metadata" poster="${prefix}assets/video/poster.webp" aria-label="75 种手绘风格的 45 秒翻书演示，含背景音乐"><source src="${prefix}assets/video/${videoName}" type="video/mp4">你的浏览器不支持视频播放，可<a href="${prefix}assets/video/${videoName}">下载 MP4</a>。</video></section>`;
}
const nav = `<nav class="site-nav" aria-label="项目导航"><a href="${repository}">GitHub 项目</a><a href="${repository}#快速开始">安装与使用</a><a href="demo/">45 秒视频</a></nav>`;
const footer = `<p class="footer">全部 327 张配图统一采用“老人和小孩过桥”场景，由本项目制作。常用精选按用途选编；个人风格收藏请在自己的 Skill 工作区中管理。<br><a href="${repository}">GitHub 项目</a> · <a href="licenses/">参考与许可</a> · <a href="licenses/BGM_CREDITS.md">音乐署名</a></p>`;
let html = sourceHtml
  .replace('</style>', `${extraCss}</style>`)
  .replace('<title>手绘主题资产库 · 327</title>', '<title>327 项手绘风格资产 · Story to Hand-Drawn Video</title><meta name="description" content="浏览 297 种手绘画风与 30 种主题配色，观看 45 秒翻书演示，选择和收藏你的故事风格。">')
  .replace('<body><main>', `<body><main>${nav}`)
  .replace('</div></header>\n<div class="controls">', `</div></header>\n${showcase()}\n<div class="controls">`)
  .replace(/<p class="footer">[\s\S]*?<\/p>/, footer)
  .replace(catalogPattern, (_, before, _json, after) => before + JSON.stringify(data).replace(/</g, '\\u003c') + after)
  .replace("link.href=item.image", "link.href=item.preview")
  .replace("byId('preview-image').src=item.image", "byId('preview-image').src=item.preview;byId('preview-original').href=item.original")
  .replace("item.name+' 原始示意图'", "item.name+' 标准配图预览'")
  .replace('<p>统一过桥场景 · 本项目标准配图</p></dialog>', '<p class="preview-links"><span>统一过桥场景 · 本项目标准配图</span><a id="preview-original" target="_blank" rel="noopener">查看原始 PNG ↗</a></p></dialog>');
if (!html.includes('<video controls playsinline preload="metadata"') || !html.includes('id="preview-original"')) fail('Gallery structure changed; static-site transformation needs an update.');
writeFileSync(resolve(output, 'index.html'), html);

const baseCss = sourceHtml.match(/<style>([\s\S]*?)<\/style>/)[1];
const page = (title, content) => `<!doctype html>\n<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${escape(title)}</title><style>${baseCss}${extraCss}</style></head><body><main class="demo-page">${content}</main></body></html>\n`;
writeFileSync(resolve(output, 'demo/index.html'), page('45 秒手绘风格翻书演示', `<nav class="site-nav"><a href="../">← 风格资产库</a><a href="${repository}">GitHub 项目</a></nav>${showcase('../', true)}<p class="footer">1080 × 1440 · 30 fps · 45 秒 · 75 种风格 · <a href="../licenses/">参考与许可</a></p>`));
writeFileSync(resolve(output, 'licenses/index.html'), page('参考与许可', `<nav class="site-nav"><a href="../">← 风格资产库</a><a href="../demo/">视频演示</a><a href="${repository}">GitHub 项目</a></nav><h1>参考与许可</h1><h2>项目与标准配图</h2><p>本项目代码按 <a href="project-MIT.txt">MIT 许可</a>公开。327 张统一“老人和小孩过桥”示意图由本项目制作，原始文件保存在 <a href="${repository}/tree/${revision}/references/style-examples/standard">GitHub 仓库</a>。</p><h2>风格与配色文字参考</h2><p>扩展风格和配色文字参考 <a href="https://github.com/yang0/handraw-style/tree/ebfeaa54953dee0fe3be01e28e6c591f30530242">yang0/handraw-style</a>，保留其 <a href="style-metadata-MIT.txt">MIT 许可声明</a>。原有风格整理参考 <a href="https://github.com/threerocks/hand-drawn-styles">hand-drawn-styles</a>，保留其<a href="original-styles-LICENSE.txt">许可声明</a>。网站使用本项目标准场景配图。</p><h2>演示视频配乐</h2><p>${credits}</p><p><a href="BGM_CREDITS.md">完整署名与编辑说明</a>。下载或转发带音乐的视频时，请保留音乐署名与许可链接。</p>`));

// Check every shipped HTML asset URL and catalog image before deployment.
for (const pagePath of ['index.html', 'demo/index.html', 'licenses/index.html']) {
  const absolutePage = resolve(output, pagePath);
  const text = readFileSync(absolutePage, 'utf8');
  if (/file:\/\/|\/Users\/|\/private\//.test(text)) fail(`Local path leaked into ${pagePath}`);
  for (const match of text.matchAll(/(?:src|href|poster)="([^"]+)"/g)) {
    const value = match[1];
    if (/^(?:https?:|#|data:)/.test(value)) continue;
    const target = resolve(dirname(absolutePage), value.split('#')[0]);
    if (!target.startsWith(output + '/') && target !== output) fail(`Path escapes site: ${value}`);
    if (!existsSync(target)) fail(`Broken local URL in ${pagePath}: ${value}`);
  }
}
for (const item of data.items) for (const key of ['image', 'preview']) if (!existsSync(resolve(output, item[key]))) fail(`Missing ${key}: ${item.id}`);
const shippedVideo = resolve(output, 'assets/video', videoName);
if (sha256(sourceVideo) !== sha256(shippedVideo)) fail('Video bytes changed during copy.');
function filesIn(dir) {return readdirSync(dir, {withFileTypes:true}).filter((entry) => entry.name !== '.vercel' && entry.name !== '.gitignore' && !entry.name.startsWith('.env')).flatMap((entry) => entry.isDirectory() ? filesIn(resolve(dir, entry.name)) : [resolve(dir, entry.name)]);}
const publicFiles = new Set(['index.html', 'demo/index.html', 'licenses/index.html', 'licenses/project-MIT.txt', 'licenses/style-metadata-MIT.txt', 'licenses/original-styles-LICENSE.txt', 'licenses/BGM_CREDITS.md', 'assets/video/poster.webp', `assets/video/${videoName}`, 'vercel.json', '.vercelignore', ...data.items.flatMap((item) => [item.image, item.preview])]);
const files = filesIn(output).filter((path) => path !== cachePath);
for (const path of files) if (!publicFiles.has(relative(output, path))) fail(`Unexpected deployment file: ${relative(output, path)}. Remove it before deploying.`);
const totalBytes = files.reduce((sum, path) => sum + statSync(path).size, 0);
if (totalBytes >= 100_000_000) fail(`Static deployment exceeds 100 MB: ${totalBytes} bytes`);
current.summary = {assets:data.items.length, featured:data.featured.length, categories:data.categories.length, thumbnailWidth:440, previewWidth:1120, videoBytes:statSync(shippedVideo).size, videoSha256:sha256(shippedVideo), totalBytes, files:files.length};
writeFileSync(cachePath, JSON.stringify(current, null, 2) + '\n');
console.log(JSON.stringify({output:relative(root, output), ...current.summary, totalMB:Number((totalBytes / 1_000_000).toFixed(2))}, null, 2));
