import {dirname, resolve} from 'node:path';
import {fileURLToPath} from 'node:url';
import {loadStyleLibrary} from './handdrawn-style-library.mjs';

const root = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const {catalog} = loadStyleLibrary(root);
const args = process.argv.slice(2);
const value = (flag, fallback = '') => {
  if (!args.includes(flag)) return fallback;
  const next = args[args.indexOf(flag) + 1];
  if (!next || next.startsWith('--')) throw new Error(`${flag} requires a value`);
  return next;
};
const type = value('--type', 'style');
if (!['style', 'palette', 'all'].includes(type)) throw new Error('--type must be style, palette or all');
const category = value('--category');
const query = value('--query').toLocaleLowerCase();
const showAll = args.includes('--all') || category || query || type !== 'style';
let items = [...(type !== 'palette' ? catalog.styles : []), ...(type !== 'style' ? catalog.palettes : [])];
if (!showAll) items = catalog.featured_styles.map((id) => items.find((s) => s.id === id));
if (category) items = items.filter((s) => [s.category, s.group, s.category_zh,
  catalog.categories.find((c) => c.id === s.category)?.name_zh].includes(category));
if (query) items = items.filter((s) => [s.id, s.name_zh, s.name_en, s.summary, ...(s.aliases || []), ...(s.best_for || [])].join(' ').toLocaleLowerCase().includes(query));
const summary = {builtin_styles: catalog.builtin_style_count, palettes: catalog.palettes.length,
  builtin_assets: catalog.builtin_style_count + catalog.palettes.length,
  custom_styles: catalog.styles.length - catalog.builtin_style_count, shown: items.length,
  default_style: catalog.default_style, selection: showAll ? 'filtered-library' : 'featured-30',
  featured_basis: catalog.featured_basis, categories: catalog.categories};
if (args.includes('--json')) console.log(JSON.stringify({summary, items}, null, 2));
else {
  console.log(`${summary.builtin_assets} built-in assets = ${summary.builtin_styles} styles + ${summary.palettes} palettes; ${summary.custom_styles} user styles`);
  console.log(`${showAll ? 'Filtered library' : '常用精选 30（按使用场景选编，非热度统计）'} · showing ${items.length}`);
  console.log('Categories: ' + catalog.categories.map((c) => `${c.id}=${c.name_zh}`).join(' / '));
  for (const item of items) {
    console.log(`\n${item.order || item.id}  ${item.id} · ${item.name_zh}${item.id === catalog.default_style ? ' ★ default' : ''}`);
    console.log(`  [${item.type}/${item.category}] ${item.summary || item.prompt}`);
    if (item.example_image) console.log(`  Example: ${item.example_image}`);
  }
}
