// Deterministic, offline adapter. Never execute code from the upstream catalog.
import {createHash} from 'node:crypto';
import {readFileSync, writeFileSync} from 'node:fs';
import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const read = (file) => JSON.parse(readFileSync(resolve(root, file), 'utf8'));
const vendor = 'references/vendor/yang0-handraw-style';
const source = read(`${vendor}/source.json`);
for (const [file, expected] of Object.entries(source.files)) {
  const actual = createHash('sha256').update(readFileSync(resolve(root, vendor, file))).digest('hex');
  if (actual !== expected) throw new Error(`Upstream snapshot checksum mismatch: ${file}`);
}
const legacy = read('references/legacy-handdrawn-styles.json');
const upstream = read(`${vendor}/styles.json`);
const previewManifest = read('references/standard-preview-manifest.json');
if (previewManifest.scene_id !== 'grandmother-child-stone-bridge-v1' || previewManifest.images.length !== 327) throw new Error('Expected 327 project standard bridge previews');
const previews = new Map();
for (const entry of previewManifest.images) {
  if (previews.has(entry.id)) throw new Error(`Duplicate preview: ${entry.id}`);
  const bytes = readFileSync(resolve(root, entry.path));
  if (createHash('sha256').update(bytes).digest('hex') !== entry.sha256) throw new Error(`Preview checksum mismatch: ${entry.path}`);
  previews.set(entry.id, entry);
}
const previewFor = (id) => {
  const entry = previews.get(id);
  if (!entry) throw new Error(`Missing local preview: ${id}`);
  return {example_image: entry.path, example_origin: {
    kind: 'project-standard', scene_id: previewManifest.scene_id,
    generator: entry.generator, sha256: entry.sha256,
  }};
};

// These upstream records have no traits. These are local interpretations of
// the technique name, not observations of upstream pictures or artist claims.
const supplements = {
  '055': '多彩蜿蜒轮廓、重叠的不规则圆形角色、稀疏亮色点染；让有机形状之间保持可读的空隙。',
  '201': '圆润短小角色、轻柔曲线、粉彩平涂与圆点五官；少量装饰保持童真。',
  '202': '细铅笔轮廓、拉长人物、薄雾般蓝绿淡彩与大留白；用远近距离表达青春心绪。',
  '203': '细墨线、清楚衣褶、柔和低饱和渐层笔触与克制表情；保留纸上漫画的线条节奏。',
  '204': '干湿交替的墨笔、大片留白、赭石与朱红小块；现代简化造型结合书写性轮廓。',
  '206': '轻快钢笔线、普通日常人物、灰蓝与暖灰淡彩；以一两件城市道具交代环境。',
  '207': '极少黑色轮廓、圆小人物与一处暖色；用人物间的距离和姿态讲述关系。',
  '208': '东方装饰曲线、疏密对比、墨黑与一处金赭点色；商业海报般明确主体与留白。',
  '209': '松软不闭合轮廓、透明水彩叠染、笨拙童趣比例；白纸间隙和浅色边缘清楚可见。',
  '210': '稚拙墨线、奇异但可读的动植物形态、少量青绿淡彩；以空白形成安静幻想空间。',
  '211': '细短铅笔线、清透蓝绿小色块、轻盈人物与极少植物纹理；保留速写的停顿。',
  '212': '略粗黑线、复古桃红与墨绿平涂、轻微网点；造型简洁、姿态松弛。',
  '213': '蓬松彩铅边缘、圆头小人物、暖黄和灰蓝；可见短笔触和柔软留白。',
  '214': '流动衣纹、夸张而清楚的剪影、青绿与朱砂薄涂；东方幻想纹样仅作小面积点缀。',
  '215': '松散日记线稿、低饱和暖色、自然人物姿势；通过少量生活物件表达温暖。',
  '216': '修长时装人物、简洁墨线、少量对比色块；用轮廓和姿态表达服装，背景留白。',
  '263': '高对比圆形与三角形符号、单眼几何角色、黑色剪影加少量原色；动作有节奏感。',
  '264': '棱角分明的粗糙钢笔线、枯枝般交叉排线、暗赭与灰色；夸张剪影表达怪诞童话。',
  '265': '二维装饰性曲线、民俗几何纹样、平涂自然色；角色轮廓与场景形成清楚节奏。',
  '269': '有力的铅笔轮廓、红黄暖色大形、明确人物姿态与版画颗粒；故事内容由当前原文决定。',
  '271': '质朴细墨线、轻水彩平涂、自然人物比例、低饱和蓝绿和暖红；旧课本印刷的轻微颗粒。',
};
const categories = [
  ['diary', '日记与生活'], ['line', '线描与讲解'], ['crayon', '蜡笔与彩铅'],
  ['ink', '水墨与国风'], ['watercolor', '水彩与淡彩'], ['gouache', '水粉与绘本'],
  ['print', '版画与印刷'], ['collage', '拼贴与纸艺'], ['graphic', '几何与平面'],
  ['comic', '漫画与幽默'], ['animation', '动画与材质'],
].map(([id, name_zh]) => ({id, name_zh}));
const classify = (value) => {
  for (const [pattern, category] of [
    [/clay|knit|3d|puppet|felt|毛毡|黏土|定格|木偶/i, 'animation'],
    [/collage|cut.paper|papercraft|paper sculpture|拼贴|剪纸|纸雕/i, 'collage'],
    [/riso|silkscreen|print|linocut|zine|版画|木刻|孔版/i, 'print'],
    [/crayon|pencil|pastel|蜡笔|彩铅|油画棒/i, 'crayon'],
    [/sumi|ink.wash|guofeng|guochao|dunhuang|水墨|国风|国潮|工笔|敦煌/i, 'ink'],
    [/watercolor|wash|水彩|淡彩/i, 'watercolor'],
    [/gouache|storybook|picturebook|水粉|绘本/i, 'gouache'],
    [/diary|notebook|lifestyle|日记|生活/i, 'diary'],
    [/geometric|flat|graphic|negative.space|几何|平面|扁平/i, 'graphic'],
    [/line|pen|ballpoint|contour|whiteboard|线描|线条|钢笔|圆珠笔|白板/i, 'line'],
  ]) if (pattern.test(value)) return category;
  return 'comic';
};
const featuredExtra = ['003', '021', '036', '045', '047', '061', '085', '088', '128', '225'];
const friendlyNames = {
  '003': '城市细线小景', '021': '水彩手写生活札记', '036': '松线淡彩绘本',
  '045': '极简诗意绘本', '047': '低饱和冷幽默绘本', '061': '中古几何童话',
  '085': '极简时尚线描', '088': '透明留白水彩', '128': '四季旅行水彩日记',
  '225': '肌理剪纸拼贴绘本',
};
const legacyCategories = ['diary', 'line', 'crayon', 'crayon', 'line', 'comic', 'line', 'crayon', 'ink', 'watercolor', 'gouache', 'gouache', 'gouache', 'watercolor', 'graphic', 'diary', 'collage', 'line', 'line', 'print'];
const styles = legacy.styles.map((style) => ({...style, ...previewFor(style.id), type: 'style',
  category: legacyCategories[style.order - 1], featured: true}));
for (const item of upstream) {
  const traits = item.traits.trim() || supplements[item.number];
  if (!traits) throw new Error(`Missing traits for HS-${item.number}`);
  const id = `hs-${item.number}`;
  const name = friendlyNames[item.number] ||
    (/^[\u3400-\u9fff]/.test(item.generation_name) ? item.generation_name : traits.split(/[；;。]/)[0].split('、').slice(0, 2).join('·'));
  styles.push({
    id, order: 20 + Number(item.number), type: 'style',
    name_zh: `${name}（HS-${item.number}）`, name_en: item.generation_name,
    aliases: [`HS-${item.number}`, `yang0:${item.number}`],
    category: classify(`${item.generation_name} ${traits}`), group: item.group,
    featured: featuredExtra.includes(item.number),
    best_for: ['故事叙事', item.group.slice(2).split(' / ')[0]], summary: traits,
    prompt_blocks: [
      `Drawing technique: ${item.generation_name}.`, traits,
      'Extract only mark-making, shapes, material, color relationships and expressive rhythm from this recipe. Example people, clothes, objects, poses and places are not a cast or scene list; the current narration alone controls content. Keep one readable moment per page.',
    ],
    caption_prompt: 'Render the exact Simplified Chinese caption as large, readable hand-drawn lettering matching the illustration medium and stroke weight. Keep natural baseline variation; do not imitate a fixed calligraphy font. No invented, missing or repeated glyphs.',
    color_hint: 'Follow the palette and material cues in the selected technique; keep clear value hierarchy, restrained accents and a light caption panel.',
    avoid: 'no unrelated drawing technique, copied sample characters or composition, stock typography, watermarks, signatures, extra text or clipped marks',
    reference_images: [], ...previewFor(id),
    origin: {kind: 'repo-adapted', url: source.repository, commit: source.commit,
      style: item.number, label: item.reference, license: 'MIT',
      traits_status: item.traits.trim() ? 'upstream' : 'local-interpretation-unverified'},
  });
}
const palettes = read(`${vendor}/colors.json`).map((item) => ({
  id: item.id, type: 'palette', category: item.category, category_zh: item.category_zh,
  name_zh: item.name_zh, name_en: item.name_en,
  ...previewFor(item.id),
  prompt: `${item.prompt_zh} ${item.prompt_en} Use this as the dominant accent family; preserve the selected medium, light caption panel and legibility.`,
  origin: {url: source.repository, commit: source.commit, license: 'MIT'},
}));
const catalog = {
  version: 2, default_style: legacy.default_style,
  description: '327 addressable recipe assets: 297 style entries and 30 theme palettes. Related techniques may overlap; these are not 327 independently verified visual families.',
  contact_sheet: legacy.contact_sheet, contact_sheet_scope: 'legacy-20-only',
  featured_basis: 'Editorial selection for common story-video use cases and medium coverage; no popularity telemetry.',
  featured_styles: [...legacy.styles.map((s) => s.id), ...featuredExtra.map((id) => `hs-${id}`)],
  categories, styles, palettes,
};
const output = `${JSON.stringify(catalog, null, 2)}\n`;
const path = resolve(root, 'references/handdrawn-style-library.json');
if (process.argv.includes('--check')) {
  if (readFileSync(path, 'utf8') !== output) throw new Error('Catalog is stale: run npm run styles:build');
} else writeFileSync(path, output);
console.log(`${styles.length} styles + ${palettes.length} palettes; ${catalog.featured_styles.length} featured`);
