import {existsSync, readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {profileToStyle, userStyles} from './reference-style.mjs';

export const styleLibraryPath = (root) => resolve(root, 'references/handdrawn-style-library.json');
const normalize = (value) => String(value || '').trim().toLocaleLowerCase();

export const loadStyleLibrary = (root, {includeUser = true} = {}) => {
  const path = styleLibraryPath(root);
  const catalog = JSON.parse(readFileSync(path, 'utf8'));
  if (!Array.isArray(catalog.styles) || !catalog.styles.length) throw new Error(`Empty style library: ${path}`);
  catalog.builtin_style_count = catalog.styles.length;
  if (includeUser) catalog.styles.push(...userStyles(root));
  const selectors = new Map();
  const ids = new Set();
  const orders = new Set();
  for (const style of catalog.styles) {
    if (!style.id || !style.name_zh || (!style.id.startsWith('custom:') && !Number.isInteger(style.order))) throw new Error(`Invalid style: ${style.id}`);
    if (ids.has(style.id) || (style.order !== undefined && orders.has(style.order))) throw new Error(`Duplicate style id/order: ${style.id}`);
    ids.add(style.id);
    if (style.order !== undefined) orders.add(style.order);
    for (const key of [style.id, style.name_zh, style.name_en, style.order === undefined ? '' : String(style.order), ...(style.aliases || [])]) {
      const normalized = normalize(key);
      if (!normalized) continue;
      const matches = selectors.get(normalized) || new Set();
      matches.add(style.id);
      selectors.set(normalized, matches);
    }
    if (!style.profile_file && (!Array.isArray(style.prompt_blocks) || !style.prompt_blocks.length || style.prompt_blocks.some((p) => typeof p !== 'string' || !p.trim()))) throw new Error(`Missing recipe for ${style.id}`);
    if (!catalog.categories.some((c) => c.id === style.category)) throw new Error(`Unknown category for ${style.id}`);
    if (style.example_image && !existsSync(resolve(root, style.example_image))) {
      style.example_image = null;
    }
  }
  if (!ids.has(catalog.default_style)) throw new Error('Missing default style');
  if (new Set(catalog.featured_styles).size !== 30 || catalog.featured_styles.some((id) => !ids.has(id))) throw new Error('Catalog must have exactly 30 valid unique featured styles');
  if (catalog.contact_sheet && !existsSync(resolve(root, catalog.contact_sheet))) {
    catalog.contact_sheet = null;
  }
  for (const palette of catalog.palettes) {
    if (palette.example_image && !existsSync(resolve(root, palette.example_image))) {
      palette.example_image = null;
    }
  }
  return {catalog, path, selectors};
};

const hydrateStyle = (root, style, library) => {
  const profilePath = style.profile_file ? resolve(root, style.profile_file) : null;
  if (profilePath && !existsSync(profilePath)) throw new Error(`Missing style profile: ${profilePath}`);
  const prompt = profilePath ? readFileSync(profilePath, 'utf8').trim() : style.prompt_blocks.join('\n');
  const references = (style.reference_images || []).map((reference) => ({...reference, absolute_path: resolve(root, reference.path)}));
  for (const reference of references) if (!existsSync(reference.absolute_path)) throw new Error(`Missing style reference: ${reference.absolute_path}`);
  return {...style, prompt, profile_path: profilePath, references,
    example_path: style.example_image ? resolve(root, style.example_image) : null,
    contact_sheet_path: library.catalog.contact_sheet ? resolve(root, library.catalog.contact_sheet) : null,
    library_path: library.path, library_version: library.catalog.version,
    is_default: style.id === library.catalog.default_style};
};

export const resolveStyle = (root, selector, {profile} = {}) => {
  const library = loadStyleLibrary(root);
  if (profile) {
    if (selector) throw new Error('Choose --style or --style-profile, not both');
    const style = profileToStyle(resolve(root, profile));
    if (!library.catalog.categories.some((c) => c.id === style.category)) throw new Error(`Unknown profile category: ${style.category}`);
    return hydrateStyle(root, style, library);
  }
  const requested = normalize(selector || library.catalog.default_style);
  const matches = library.selectors.get(requested);
  if (!matches) throw new Error(`Unknown --style "${selector}". Use npm run styles -- --all; upstream numbers use HS-001 through HS-277.`);
  const exact = library.catalog.styles.find((item) => normalize(item.id) === requested);
  if (!exact && matches.size > 1) throw new Error(`Ambiguous style "${selector}"; use one id: ${[...matches].join(', ')}`);
  return hydrateStyle(root, exact || library.catalog.styles.find((item) => item.id === [...matches][0]), library);
};

export const orderedStyles = (root) => [...loadStyleLibrary(root).catalog.styles].sort((a, b) => (a.order || 10000) - (b.order || 10000));

export const resolvePalette = (root, selector) => {
  if (!selector) return null;
  const {catalog} = loadStyleLibrary(root, {includeUser: false});
  const palette = catalog.palettes.find((p) => [p.id, p.name_zh, p.name_en].some((key) => normalize(key) === normalize(selector)));
  if (!palette) throw new Error(`Unknown --palette "${selector}". Use npm run styles -- --type palette`);
  return palette;
};
