import {execFileSync} from 'node:child_process';
import {copyFileSync} from 'node:fs';
import {resolve} from 'node:path';

export const dimensionsFor = (path) => {
  const output = execFileSync(
    'ffprobe',
    [
      '-v',
      'error',
      '-select_streams',
      'v:0',
      '-show_entries',
      'stream=width,height',
      '-of',
      'json',
      path,
    ],
    {encoding: 'utf8'},
  );
  const stream = JSON.parse(output).streams?.[0];
  if (!stream?.width || !stream?.height) {
    throw new Error(`Could not read image dimensions: ${path}`);
  }
  return {width: Number(stream.width), height: Number(stream.height)};
};

export const analyzeCompositeLayout = (path, width, height) => {
  const previewWidth = 256;
  const pixels = execFileSync(
    'ffmpeg',
    [
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      path,
      '-vf',
      `scale=${previewWidth}:-2:flags=area,format=gray`,
      '-frames:v',
      '1',
      '-f',
      'rawvideo',
      '-pix_fmt',
      'gray',
      '-',
    ],
    {maxBuffer: 8 * 1024 * 1024},
  );
  const previewHeight = Math.floor(pixels.length / previewWidth);
  const rowInk = Array.from({length: previewHeight}, (_, y) => {
    let ink = 0;
    const offset = y * previewWidth;
    for (let x = 0; x < previewWidth; x += 1) {
      if (pixels[offset + x] < 238) ink += 1;
    }
    return ink / previewWidth;
  });
  const smoothed = rowInk.map((_, y) => {
    let total = 0;
    let count = 0;
    for (let offset = -2; offset <= 2; offset += 1) {
      const row = y + offset;
      if (row >= 0 && row < previewHeight) {
        total += rowInk[row];
        count += 1;
      }
    }
    return total / count;
  });

  const searchStart = Math.round(previewHeight * 0.22);
  const searchEnd = Math.round(previewHeight * 0.52);
  const runs = [];
  let runStart = null;
  for (let y = searchStart; y <= searchEnd; y += 1) {
    if (smoothed[y] < 0.012 && runStart === null) runStart = y;
    const closes = smoothed[y] >= 0.012 || y === searchEnd;
    if (closes && runStart !== null) {
      const end = smoothed[y] >= 0.012 ? y - 1 : y;
      runs.push({start: runStart, end, length: end - runStart + 1});
      runStart = null;
    }
  }
  runs.sort((a, b) => b.length - a.length || a.start - b.start);
  const bestRun = runs[0] || null;
  let splitPreview = bestRun
    ? Math.round((bestRun.start + bestRun.end) / 2)
    : searchStart;
  if (!bestRun) {
    for (let y = searchStart; y <= searchEnd; y += 1) {
      if (smoothed[y] < smoothed[splitPreview]) splitPreview = y;
    }
  }

  // Some uploaded diary-comic pages leave a wide white gutter immediately
  // after a short one-line caption. Clamping the split to 27% pushed the crop
  // down into roofs, hair, or a third caption line on those pages. The search
  // already starts at 22%, so allow the detected gutter itself to determine
  // the boundary while retaining the existing upper safety limit.
  const minSplit = searchStart;
  const maxSplit = Math.round(previewHeight * 0.5);
  splitPreview = Math.max(minSplit, Math.min(maxSplit, splitPreview));

  const contentRows = [];
  for (let y = 0; y < splitPreview; y += 1) {
    if (rowInk[y] > 0.012) contentRows.push(y);
  }
  const scaleY = height / previewHeight;
  const detectedCaption =
    contentRows.length > Math.max(4, previewHeight * 0.02) &&
    Boolean(bestRun && bestRun.length >= previewHeight * 0.012);
  const topContent = contentRows[0] ?? 0;
  const bottomContent = contentRows.at(-1) ?? splitPreview;
  const padding = Math.max(8, Math.round(previewHeight * 0.018));
  const captionY = Math.max(0, Math.round((topContent - padding) * scaleY));
  const captionBottom = Math.min(
    height,
    Math.round((bottomContent + padding) * scaleY),
  );

  return {
    hasCaption: detectedCaption,
    splitY: Math.round(splitPreview * scaleY),
    captionY,
    captionH: Math.max(24, captionBottom - captionY),
  };
};

export const runFfmpeg = (input, filter, output) => {
  execFileSync(
    'ffmpeg',
    [
      '-hide_banner',
      '-loglevel',
      'error',
      '-i',
      input,
      '-vf',
      filter,
      '-frames:v',
      '1',
      '-y',
      output,
    ],
    {stdio: 'inherit'},
  );
};

// Both uploaded pages and generated lettering use this processor. Derive BW
// from the final color plate to keep every pixel aligned. Preserve masters.
export const processPage = ({input, masterPath, textPath, colorPath, bwPath,
  transition = 'cut', layout = 'auto', splitOverride, generatedTextMode}) => {
  const {width, height} = dimensionsFor(input);
  if (!['cut', 'page-flip'].includes(transition)) throw new Error('Invalid page transition');
  let hasCaption = false;
  let splitY = 0;
  let captionY = 0;
  let captionH = 0;
  if (generatedTextMode === 'image2') {
    if (Math.abs(width / height - 2 / 3) > 0.015) {
      throw new Error(`Generated caption master must be 2:3, got ${width}x${height}: ${input}`);
    }
    // Image models do not reliably honor pixel coordinates in a prompt. Use
    // the same observed whitespace/ink geometry as uploaded pages, so a
    // drawing starting above y=512 is not accidentally cut off.
    if (transition !== 'page-flip') {
      const detection = analyzeCompositeLayout(input, width, height);
      if (!detection.hasCaption) throw new Error(`No clear caption/art separation found; inspect or regenerate the master: ${input}`);
      hasCaption = true;
      splitY = detection.splitY;
      captionY = detection.captionY;
      captionH = Math.max(1, Math.min(detection.captionH, splitY - captionY));
    }
  } else if (!generatedTextMode && transition !== 'page-flip' && layout !== 'full') {
    const detection = analyzeCompositeLayout(input, width, height);
    hasCaption = layout === 'composite' || detection.hasCaption;
    if (splitOverride !== undefined && (!Number.isInteger(splitOverride) ||
      splitOverride < Math.round(height * 0.16) || splitOverride > Math.round(height * 0.62))) {
      throw new Error('--split-y is outside the safe range');
    }
    splitY = hasCaption ? splitOverride ?? detection.splitY : 0;
    captionY = splitOverride === undefined ? detection.captionY : 0;
    captionH = splitOverride === undefined ? detection.captionH : splitOverride;
    // Never let a text crop cross the illustration boundary.
    captionH = Math.max(1, Math.min(captionH, splitY - captionY));
  }
  if (resolve(input) !== resolve(masterPath)) copyFileSync(input, masterPath);
  if (transition === 'page-flip') return {width, height, hasCaption, splitY, captionY, captionH};
  if (hasCaption) {
    runFfmpeg(masterPath,
      `crop=${width}:${captionH}:0:${captionY},scale=1536:512:force_original_aspect_ratio=decrease:flags=lanczos,pad=1536:512:(ow-iw)/2:(oh-ih)/2:color=white`, textPath);
  }
  const crop = hasCaption ? `crop=${width}:${height - splitY}:0:${splitY},` : '';
  runFfmpeg(masterPath,
    `${crop}scale=1024:1024:force_original_aspect_ratio=decrease:flags=lanczos,pad=1024:1024:(ow-iw)/2:(oh-ih)/2:color=white`, colorPath);
  runFfmpeg(colorPath, 'format=gray,eq=contrast=1.18:brightness=0.035,unsharp=5:5:0.55:5:5:0', bwPath);
  return {width, height, hasCaption, splitY, captionY, captionH};
};
