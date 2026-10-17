const L = require("./learn.js");

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

function almost(a, b, eps, msg) {
  if (Math.abs(a - b) > eps) throw new Error(`${msg}: ${a} !== ${b}`);
}

function blank(w, h, rgb) {
  const data = new Uint8ClampedArray(w * h * 4);
  for (let i = 0; i < w * h; i++) {
    data[i * 4] = rgb[0];
    data[i * 4 + 1] = rgb[1];
    data[i * 4 + 2] = rgb[2];
    data[i * 4 + 3] = 255;
  }
  return { width: w, height: h, data };
}

function paint(img, x, y, rgb) {
  if (x < 0 || y < 0 || x >= img.width || y >= img.height) return;
  const i = (y * img.width + x) * 4;
  img.data[i] = rgb[0];
  img.data[i + 1] = rgb[1];
  img.data[i + 2] = rgb[2];
  img.data[i + 3] = 255;
}

function chartFromPrices(prices, opts) {
  const o = opts || {};
  const w = o.w || 180;
  const h = o.h || 110;
  const padX = o.padX == null ? 8 : o.padX;
  const y0 = o.y0 == null ? 8 : o.y0;
  const y1 = o.y1 == null ? h - 10 : o.y1;
  const img = blank(w, h, o.bg || [11, 14, 17]);
  const min = Math.min.apply(null, prices);
  const max = Math.max.apply(null, prices);
  const span = max - min || 1;
  for (let i = 0; i < prices.length; i++) {
    const x = padX + Math.round((i / (prices.length - 1)) * (w - padX * 2 - 2));
    const norm = (prices[i] - min) / span;
    const y = y0 + (1 - norm) * (y1 - y0);
    const up = i === 0 || prices[i] >= prices[i - 1];
    const rgb = up ? [8, 153, 129] : [242, 54, 69];
    for (let dy = -1; dy <= 5; dy++) {
      for (let dx = 0; dx <= 1; dx++) paint(img, x + dx, Math.round(y) + dy, rgb);
    }
  }
  if (o.volume) {
    for (let i = 0; i < prices.length; i++) {
      const x = padX + Math.round((i / (prices.length - 1)) * (w - padX * 2 - 2));
      const vh = 4 + ((i * 17) % 12);
      const up = i === 0 || prices[i] >= prices[i - 1];
      const rgb = up ? [8, 153, 129] : [242, 54, 69];
      for (let dy = 0; dy < vh; dy++) {
        paint(img, x, h - 4 - dy, rgb);
        paint(img, x + 1, h - 4 - dy, rgb);
      }
    }
  }
  return img;
}

function reclaim(n) {
  const p = [];
  const turn = Math.floor(n * 0.62);
  for (let i = 0; i < n; i++) {
    if (i <= turn) p.push(100 - i * 2.1);
    else p.push(100 - turn * 2.1 + (i - turn) * 4.6);
  }
  return p;
}

function grindDown(n) {
  const p = [];
  for (let i = 0; i < n; i++) p.push(100 - i * 2.2);
  return p;
}

function grindUp(n) {
  const p = [];
  for (let i = 0; i < n; i++) p.push(20 + i * 2);
  return p;
}

function pearson(a, b) {
  const n = Math.min(a.length, b.length);
  const ma = a.slice(0, n).reduce((s, v) => s + v, 0) / n;
  const mb = b.slice(0, n).reduce((s, v) => s + v, 0) / n;
  let num = 0;
  let da = 0;
  let db = 0;
  for (let i = 0; i < n; i++) {
    const xa = a[i] - ma;
    const xb = b[i] - mb;
    num += xa * xb;
    da += xa * xa;
    db += xb * xb;
  }
  return num / Math.sqrt(da * db);
}

const flip = L.extractChartFeatures(chartFromPrices(reclaim(36)));
assert(!flip.unreadable, "reclaim should be readable");
assert(flip.metrics.drop > 0.6, "reclaim drops");
assert(flip.metrics.recovery > 0.45, "reclaim recovers");
assert(flip.metrics.flipAfter === 1, "reclaim is flip-after-break");
assert(flip.shapeScore > 0.7, `prior likes reclaim, got ${flip.shapeScore}`);
assert(flip.cues.some((c) => c.ok && c.text.includes("破底")), "cue mentions break");
assert(flip.cues.some((c) => c.ok && c.text.includes("翻")), "cue mentions flip");

const down = L.extractChartFeatures(chartFromPrices(grindDown(36)));
assert(!down.unreadable, "decline readable");
assert(down.shapeScore < 0.35, `prior rejects still-falling, got ${down.shapeScore}`);
assert(down.cues.some((c) => !c.ok && c.text.includes("還沒翻")), "cue says not flipped");

const up = L.extractChartFeatures(chartFromPrices(grindUp(36)));
assert(!up.unreadable, "uptrend readable");
assert(up.metrics.noBreak === 1 || up.metrics.earlyLow === 1 || up.metrics.drop < 0.32, "uptrend is not a breakdown");
assert(up.shapeScore < 0.45, `prior rejects uptrend, got ${up.shapeScore}`);

const withVol = L.extractChartFeatures(chartFromPrices(reclaim(36), {
  h: 140,
  y0: 8,
  y1: 78,
  volume: true
}));
assert(!withVol.unreadable, "volume chart readable");
const resampled = [];
const src = reclaim(36);
for (let i = 0; i < withVol.path.length; i++) {
  const t = (i / (withVol.path.length - 1)) * (src.length - 1);
  const j = Math.floor(t);
  const f = t - j;
  resampled.push(src[j] + (src[Math.min(src.length - 1, j + 1)] - src[j]) * f);
}
const corr = pearson(withVol.path, resampled);
assert(corr > 0.8, `price pane ignores volume, corr ${corr}`);

const empty = L.extractChartFeatures(blank(80, 60, [11, 14, 17]));
assert(empty.unreadable, "blank is unreadable");

const model = L.createModel();
for (let i = 0; i < 6; i++) {
  const series = reclaim(34 + (i % 3)).map((v, idx) => v + ((idx + i) % 3) - 1);
  const feat = L.extractChartFeatures(chartFromPrices(series, { h: 120, y0: 10, y1: 100 }));
  assert(!feat.unreadable, "train reclaim readable");
  L.addExample(model, { label: 1, note: "破底後翻上", vector: feat.vector, name: `yes-${i}` });
}
for (let i = 0; i < 6; i++) {
  const series = grindDown(34 + (i % 3)).map((v, idx) => v + (idx % 2));
  const feat = L.extractChartFeatures(chartFromPrices(series, { h: 120, y0: 10, y1: 100 }));
  assert(!feat.unreadable, "train decline readable");
  L.addExample(model, { label: 0, note: "只跌不翻", vector: feat.vector, name: `no-${i}` });
}

const holdYes = L.extractChartFeatures(chartFromPrices(reclaim(40), { h: 120, y0: 12, y1: 102 }));
const holdNo = L.extractChartFeatures(chartFromPrices(grindDown(40), { h: 120, y0: 12, y1: 102 }));
const predYes = L.predict(model, holdYes);
const predNo = L.predict(model, holdNo);
assert(predYes.status === "yes", `held-out reclaim should be yes, got ${predYes.status} ${predYes.prob}`);
assert(predNo.status === "no", `held-out decline should be no, got ${predNo.status} ${predNo.prob}`);
assert(predYes.prob > predNo.prob + 0.2, "yes outranks no");
assert(predYes.neighbors.length > 0, "neighbors returned");
assert(predYes.neighbors[0].label === 1, "nearest neighbor is a positive");
assert(predYes.counts.pos === 6 && predYes.counts.neg === 6, "counts");

const dumped = L.exportState(model, "口訣");
const revived = L.createModel();
const added = L.importExamples(revived, dumped);
assert(added === 12, "import count");
const again = L.predict(revived, holdYes);
assert(again.status === "yes", "imported model still says yes");

L.removeExample(revived, revived.examples[0].id);
assert(revived.examples.length === 11, "delete one");

const bad = L.createModel();
let threw = false;
try { L.addExample(bad, { label: 1, vector: [1, 2, 3] }); } catch (err) { threw = true; }
assert(threw, "reject short vector");

assert(L.DEFAULT_RULES.includes("破底翻"), "default rules");
assert(L.FEATURE_DIM > 40, "feature dim");

console.log("podifan learn tests ok");
console.log({
  flip: Number(flip.shapeScore.toFixed(3)),
  down: Number(down.shapeScore.toFixed(3)),
  up: Number(up.shapeScore.toFixed(3)),
  volCorr: Number(corr.toFixed(3)),
  predYes: Number(predYes.prob.toFixed(3)),
  predNo: Number(predNo.prob.toFixed(3))
});
