(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.PodifanLearn = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  var COLS = 32;
  var PATH_N = 16;
  var MASK_W = 8;
  var MASK_H = 6;

  var I = {
    PATH: 0,
    DELTA: PATH_N,
    POS: PATH_N + (PATH_N - 1),
    DROP: PATH_N + (PATH_N - 1) + 1,
    RECOVERY: PATH_N + (PATH_N - 1) + 2,
    END: PATH_N + (PATH_N - 1) + 3,
    SWEET: PATH_N + (PATH_N - 1) + 4,
    FLIPPED: PATH_N + (PATH_N - 1) + 5,
    BROKE: PATH_N + (PATH_N - 1) + 6,
    STILL_FALLING: PATH_N + (PATH_N - 1) + 7,
    EARLY_LOW: PATH_N + (PATH_N - 1) + 8,
    NO_BREAK: PATH_N + (PATH_N - 1) + 9,
    RIGHT_MINUS_LEFT: PATH_N + (PATH_N - 1) + 10,
    FLIP_AFTER_BREAK: PATH_N + (PATH_N - 1) + 11,
    MASK: PATH_N + (PATH_N - 1) + 12
  };
  var FEATURE_DIM = I.MASK + MASK_W * MASK_H;

  var DEFAULT_RULES = [
    "破底翻先這樣看：",
    "1. 先跌破一段前低，跌深要看得出來，不是小回。",
    "2. 破完之後價格翻上去，收復剛跌破的位置或短均線。",
    "3. 只跌不翻，或低點就貼在最右邊、還沒翻，先不當。",
    "你標過的圖會蓋過這段口訣。"
  ].join("\n");

  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }

  function luma(r, g, b) {
    return 0.2126 * r + 0.7152 * g + 0.0722 * b;
  }

  function saturation(r, g, b) {
    var max = Math.max(r, g, b);
    var min = Math.min(r, g, b);
    if (max === 0) return 0;
    return (max - min) / max;
  }

  function pixelAt(data, w, x, y) {
    var i = (y * w + x) * 4;
    return { r: data[i], g: data[i + 1], b: data[i + 2] };
  }

  function colorDist(p, bg) {
    var dr = p.r - bg.r;
    var dg = p.g - bg.g;
    var db = p.b - bg.b;
    return Math.sqrt(dr * dr + dg * dg + db * db);
  }

  function downscale(image, maxW) {
    var w0 = image.width;
    var h0 = image.height;
    if (!w0 || !h0) return { width: 0, height: 0, data: new Uint8ClampedArray(0) };
    if (w0 <= maxW) return image;
    var scale = maxW / w0;
    var w = maxW;
    var h = Math.max(1, Math.round(h0 * scale));
    var out = new Uint8ClampedArray(w * h * 4);
    for (var y = 0; y < h; y++) {
      var sy = Math.min(h0 - 1, Math.floor(y / scale));
      for (var x = 0; x < w; x++) {
        var sx = Math.min(w0 - 1, Math.floor(x / scale));
        var si = (sy * w0 + sx) * 4;
        var di = (y * w + x) * 4;
        out[di] = image.data[si];
        out[di + 1] = image.data[si + 1];
        out[di + 2] = image.data[si + 2];
        out[di + 3] = 255;
      }
    }
    return { width: w, height: h, data: out };
  }

  function estimateBackground(image) {
    var bins = new Map();
    var stepX = Math.max(1, Math.floor(image.width / 36));
    var stepY = Math.max(1, Math.floor(image.height / 36));
    for (var y = 0; y < image.height; y += stepY) {
      for (var x = 0; x < image.width; x += stepX) {
        var p = pixelAt(image.data, image.width, x, y);
        var key = (p.r >> 4) + "," + (p.g >> 4) + "," + (p.b >> 4);
        var hit = bins.get(key);
        if (!hit) bins.set(key, { n: 1, r: p.r, g: p.g, b: p.b });
        else {
          hit.n += 1;
          hit.r += p.r;
          hit.g += p.g;
          hit.b += p.b;
        }
      }
    }
    var best = null;
    bins.forEach(function (hit) {
      if (!best || hit.n > best.n) best = hit;
    });
    if (!best) return { r: 0, g: 0, b: 0 };
    return { r: best.r / best.n, g: best.g / best.n, b: best.b / best.n };
  }

  function isMark(p, bg) {
    var d = colorDist(p, bg);
    if (d < 36) return false;
    if (saturation(p.r, p.g, p.b) >= 0.35) return true;
    if (Math.abs(luma(p.r, p.g, p.b) - luma(bg.r, bg.g, bg.b)) >= 55) return true;
    return false;
  }

  function markGrid(image, bg) {
    var w = image.width;
    var h = image.height;
    var mark = new Uint8Array(w * h);
    var rows = new Array(h).fill(0);
    var total = 0;
    for (var y = 0; y < h; y++) {
      for (var x = 0; x < w; x++) {
        if (isMark(pixelAt(image.data, w, x, y), bg)) {
          mark[y * w + x] = 1;
          rows[y] += 1;
          total += 1;
        }
      }
    }
    return { mark: mark, rows: rows, total: total };
  }

  function findPricePane(rows) {
    var h = rows.length;
    var inked = rows.map(function (c) { return c >= 2; });
    var first = -1;
    var last = -1;
    for (var y = 0; y < h; y++) {
      if (!inked[y]) continue;
      if (first < 0) first = y;
      last = y;
    }
    if (first < 0) return null;

    var best = null;
    var runStart = -1;
    for (var yy = 0; yy <= h; yy++) {
      var on = yy < h && inked[yy];
      if (!on && runStart < 0) runStart = yy;
      if ((on || yy === h) && runStart >= 0) {
        var runEnd = yy;
        var len = runEnd - runStart;
        var above = 0;
        var below = 0;
        var a;
        for (a = 0; a < runStart; a++) above += rows[a];
        for (a = runEnd; a < h; a++) below += rows[a];
        var mid = (runStart + runEnd) / 2;
        if (len >= 8 && mid > h * 0.35 && above > 0 && below > 0) {
          if (!best || len > best.len || (len === best.len && runStart > best.start)) {
            best = { start: runStart, end: runEnd, len: len };
          }
        }
        runStart = -1;
      }
    }

    var y0 = first;
    var y1 = best ? best.start : last + 1;
    if (y1 - y0 < 8) return null;
    return { y0: y0, y1: y1 };
  }

  function resample(values, n) {
    if (!values.length) return new Array(n).fill(0);
    if (values.length === 1) return new Array(n).fill(values[0]);
    var out = new Array(n);
    for (var i = 0; i < n; i++) {
      var t = (i / (n - 1)) * (values.length - 1);
      var j = Math.floor(t);
      var f = t - j;
      var a = values[j];
      var b = values[Math.min(values.length - 1, j + 1)];
      out[i] = a + (b - a) * f;
    }
    return out;
  }

  function fillGaps(series) {
    var known = [];
    var i;
    for (i = 0; i < series.length; i++) {
      if (series[i] != null && !isNaN(series[i])) known.push(i);
    }
    if (!known.length) return series;
    var k;
    for (k = 0; k < known[0]; k++) series[k] = series[known[0]];
    for (var n = 0; n < known.length - 1; n++) {
      var a = known[n];
      var b = known[n + 1];
      for (k = a + 1; k < b; k++) {
        var t = (k - a) / (b - a);
        series[k] = series[a] + (series[b] - series[a]) * t;
      }
    }
    var tail = series[known[known.length - 1]];
    for (k = known[known.length - 1] + 1; k < series.length; k++) series[k] = tail;
    return series;
  }

  function argMin(values) {
    var idx = 0;
    for (var i = 1; i < values.length; i++) if (values[i] < values[idx]) idx = i;
    return idx;
  }

  function mean(values) {
    if (!values.length) return 0;
    var s = 0;
    for (var i = 0; i < values.length; i++) s += values[i];
    return s / values.length;
  }

  function priorModel() {
    var w = new Array(FEATURE_DIM).fill(0);
    w[I.RECOVERY] = 1.4;
    w[I.DROP] = 0.8;
    w[I.SWEET] = 0.5;
    w[I.FLIPPED] = 0.4;
    w[I.BROKE] = 0.3;
    w[I.STILL_FALLING] = -2.4;
    w[I.EARLY_LOW] = -1.6;
    w[I.NO_BREAK] = -1.3;
    w[I.FLIP_AFTER_BREAK] = 3.4;
    w[I.END] = 0.3;
    return { weights: w, bias: -1.7 };
  }

  function dot(w, x) {
    var s = 0;
    for (var i = 0; i < w.length; i++) s += w[i] * x[i];
    return s;
  }

  function sigmoid(z) {
    if (z > 24) return 1;
    if (z < -24) return 0;
    return 1 / (1 + Math.exp(-z));
  }

  function emptyRead(reason) {
    return {
      vector: new Array(FEATURE_DIM).fill(0),
      path: [],
      unreadable: true,
      shapeScore: 0,
      metrics: null,
      cues: [{ ok: false, text: reason }]
    };
  }

  function extractChartFeatures(image) {
    if (!image || !image.width || !image.height || !image.data) {
      return emptyRead("這張圖讀不到。");
    }
    var small = downscale(image, 220);
    var bg = estimateBackground(small);
    var grid = markGrid(small, bg);
    if (grid.total < small.width * small.height * 0.004) {
      return emptyRead("這張圖裡幾乎沒有 K 線或價格線，換一張截圖。");
    }
    var pane = findPricePane(grid.rows);
    if (!pane) return emptyRead("價格區分不出來，換一張 K 線截圖。");

    var bands = new Array(COLS);
    var known = 0;
    for (var c = 0; c < COLS; c++) {
      var x0 = Math.floor((c / COLS) * small.width);
      var x1 = Math.max(x0 + 1, Math.floor(((c + 1) / COLS) * small.width));
      var top = null;
      var bot = null;
      for (var y = pane.y0; y < pane.y1; y++) {
        for (var x = x0; x < x1; x++) {
          if (!grid.mark[y * small.width + x]) continue;
          if (top == null || y < top) top = y;
          if (bot == null || y > bot) bot = y;
        }
      }
      if (top == null) {
        bands[c] = null;
        continue;
      }
      var span = (bot - top) / Math.max(1, pane.y1 - pane.y0);
      if (span > 0.92) {
        bands[c] = null;
        continue;
      }
      var price = 1 - (bot - pane.y0) / Math.max(1, pane.y1 - pane.y0 - 1);
      bands[c] = clamp(price, 0, 1);
      known += 1;
    }
    if (known < COLS * 0.45) {
      return emptyRead("K 線不夠清楚，價格走勢讀不出來。");
    }
    fillGaps(bands);
    var path = resample(bands, PATH_N);
    var minI = argMin(path);
    var minV = path[minI];
    var maxV = path[0];
    for (var i = 1; i < path.length; i++) if (path[i] > maxV) maxV = path[i];
    var range = maxV - minV;
    if (range < 0.04) {
      return emptyRead("這段幾乎走平，看不出破底或翻。");
    }

    var pos = minI / (path.length - 1);
    var leftMax = minV;
    for (var L = 0; L < minI; L++) if (path[L] > leftMax) leftMax = path[L];
    var drop = minI === 0 ? 0 : (leftMax - minV) / range;
    var end = path[path.length - 1];
    var recovery = (end - minV) / range;
    var stillFalling = pos > 0.82 && recovery < 0.28 ? 1 : 0;
    var earlyLow = pos < 0.18 ? 1 : 0;
    var sweet = pos >= 0.22 && pos <= 0.78 ? 1 : 0;
    var flipped = recovery >= 0.34 ? 1 : 0;
    var broke = drop >= 0.42 ? 1 : 0;
    var noBreak = drop < 0.32 ? 1 : 0;
    var flipAfter = broke && flipped && sweet && !stillFalling ? 1 : 0;
    var leftMean = mean(path.slice(0, Math.max(1, Math.floor(path.length / 3))));
    var rightMean = mean(path.slice(Math.floor((path.length * 2) / 3)));

    var vector = new Array(FEATURE_DIM).fill(0);
    for (var p = 0; p < PATH_N; p++) vector[I.PATH + p] = path[p];
    for (var d = 0; d < PATH_N - 1; d++) vector[I.DELTA + d] = path[d + 1] - path[d];
    vector[I.POS] = pos;
    vector[I.DROP] = clamp(drop, 0, 1);
    vector[I.RECOVERY] = clamp(recovery, 0, 1.2);
    vector[I.END] = end;
    vector[I.SWEET] = sweet;
    vector[I.FLIPPED] = flipped;
    vector[I.BROKE] = broke;
    vector[I.STILL_FALLING] = stillFalling;
    vector[I.EARLY_LOW] = earlyLow;
    vector[I.NO_BREAK] = noBreak;
    vector[I.RIGHT_MINUS_LEFT] = clamp(rightMean - leftMean, -1, 1);
    vector[I.FLIP_AFTER_BREAK] = flipAfter;

    for (var my = 0; my < MASK_H; my++) {
      for (var mx = 0; mx < MASK_W; mx++) {
        var sx0 = Math.floor((mx / MASK_W) * small.width);
        var sx1 = Math.max(sx0 + 1, Math.floor(((mx + 1) / MASK_W) * small.width));
        var sy0 = pane.y0 + Math.floor((my / MASK_H) * (pane.y1 - pane.y0));
        var sy1 = pane.y0 + Math.max(1, Math.floor(((my + 1) / MASK_H) * (pane.y1 - pane.y0)));
        var hit = 0;
        var seen = 0;
        for (var yy = sy0; yy < sy1; yy++) {
          for (var xx = sx0; xx < sx1; xx++) {
            seen += 1;
            hit += grid.mark[yy * small.width + xx] ? 1 : 0;
          }
        }
        vector[I.MASK + my * MASK_W + mx] = seen ? hit / seen : 0;
      }
    }

    var metrics = {
      pos: pos,
      drop: drop,
      recovery: recovery,
      end: end,
      flipAfter: flipAfter
    };
    return {
      vector: vector,
      path: path,
      unreadable: false,
      shapeScore: shapeScoreOf(vector),
      metrics: metrics,
      cues: buildCues(metrics)
    };
  }

  function buildCues(m) {
    var cues = [];
    if (m.drop >= 0.42) {
      cues.push({ ok: true, text: "有破底：低點比左邊高點少了一大段（約 " + Math.round(m.drop * 100) + "% 的這段波幅）。" });
    } else if (m.drop >= 0.32) {
      cues.push({ ok: false, text: "有往下破，但深度普通（約 " + Math.round(m.drop * 100) + "%），還不算明顯破底。" });
    } else if (m.pos < 0.18) {
      cues.push({ ok: false, text: "最低點太靠左，比較像一開始就在低檔，不是這一段裡破底。" });
    } else {
      cues.push({ ok: false, text: "沒有明顯破底，左邊缺少一段跌深。" });
    }

    if (m.pos > 0.82 && m.recovery < 0.28) {
      cues.push({ ok: false, text: "低點貼在最右邊，價格還沒翻上去。" });
    } else if (m.recovery >= 0.34) {
      cues.push({ ok: true, text: "低點之後有翻，收復了這段跌幅的約 " + Math.round(m.recovery * 100) + "%。" });
    } else {
      cues.push({ ok: false, text: "低點之後只彈了一點（約 " + Math.round(m.recovery * 100) + "%），翻得不夠。" });
    }

    if (m.pos >= 0.22 && m.pos <= 0.78) {
      cues.push({ ok: true, text: "低點落在圖的中段，左邊有跌、右邊有空間翻。" });
    } else if (m.pos < 0.22) {
      cues.push({ ok: false, text: "低點太早出現，後面比較像漲完，不像剛破底翻。" });
    }

    if (m.flipAfter) {
      cues.push({ ok: true, text: "形狀符合：先破底，再從低點翻上去。" });
    }
    return cues;
  }

  function shapeScoreOf(vector) {
    var prior = priorModel();
    return sigmoid(dot(prior.weights, vector) + prior.bias);
  }

  function cosine(a, b) {
    var d = dot(a, b);
    var na = Math.sqrt(dot(a, a));
    var nb = Math.sqrt(dot(b, b));
    if (na === 0 || nb === 0) return 0;
    return d / (na * nb);
  }

  function createModel() {
    var prior = priorModel();
    return {
      version: 1,
      examples: [],
      weights: prior.weights.slice(),
      bias: prior.bias,
      trained: false
    };
  }

  function refit(model) {
    var prior = priorModel();
    var w = prior.weights.slice();
    var b = prior.bias;
    var examples = model.examples || [];
    model.weights = w;
    model.bias = b;
    model.trained = examples.length > 0;
    if (!examples.length) return model;

    var pos = 0;
    for (var i = 0; i < examples.length; i++) if (examples[i].label) pos += 1;
    var neg = examples.length - pos;
    var lr = 0.28;
    var lambda = examples.length < 8 ? 0.55 : 0.12;
    var epochs = 60;

    for (var ep = 0; ep < epochs; ep++) {
      for (var n = 0; n < examples.length; n++) {
        var ex = examples[n];
        var y = ex.label ? 1 : 0;
        var x = ex.vector;
        if (!x || x.length !== FEATURE_DIM) continue;
        var p = sigmoid(dot(w, x) + b);
        var cw = 1;
        if (pos > 0 && neg > 0) cw = y ? neg / pos : pos / neg;
        var err = (y - p) * cw;
        for (var k = 0; k < FEATURE_DIM; k++) w[k] += lr * err * x[k];
        b += lr * err;
      }
      for (var j = 0; j < FEATURE_DIM; j++) w[j] -= lr * lambda * (w[j] - prior.weights[j]);
      b -= lr * lambda * (b - prior.bias);
    }
    model.weights = w;
    model.bias = b;
    return model;
  }

  function roundVector(vector) {
    var out = new Array(vector.length);
    for (var i = 0; i < vector.length; i++) out[i] = Math.round(vector[i] * 10000) / 10000;
    return out;
  }

  function addExample(model, example) {
    if (!example || !example.vector || example.vector.length !== FEATURE_DIM) {
      throw new Error("教材缺少特徵");
    }
    var row = {
      id: example.id || ("ex_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8)),
      label: example.label ? 1 : 0,
      note: String(example.note || "").slice(0, 280),
      name: String(example.name || "").slice(0, 120),
      createdAt: example.createdAt || new Date().toISOString(),
      source: example.source || "local",
      vector: roundVector(example.vector),
      thumb: example.thumb || ""
    };
    model.examples.push(row);
    refit(model);
    return row;
  }

  function removeExample(model, id) {
    model.examples = model.examples.filter(function (ex) { return ex.id !== id; });
    refit(model);
  }

  function neighborVote(model, vector) {
    var ranked = model.examples.map(function (ex) {
      return { example: ex, sim: cosine(vector, ex.vector) };
    }).sort(function (a, b) { return b.sim - a.sim; });
    var top = ranked.slice(0, 3).filter(function (row) { return row.sim > 0.2; });
    if (!top.length) return { score: null, neighbors: [] };
    var num = 0;
    var den = 0;
    for (var i = 0; i < top.length; i++) {
      var weight = Math.max(0, top[i].sim);
      num += weight * (top[i].example.label ? 1 : 0);
      den += weight;
    }
    return { score: den ? num / den : null, neighbors: top };
  }

  function predict(model, extracted) {
    var vector = extracted.vector;
    var prior = priorModel();
    var priorProb = sigmoid(dot(prior.weights, vector) + prior.bias);
    var learnedProb = sigmoid(dot(model.weights, vector) + model.bias);
    var vote = neighborVote(model, vector);
    var count = model.examples.length;
    var prob = count ? learnedProb : priorProb;
    if (vote.score != null && count >= 4) {
      var mix = Math.min(0.5, count / 24);
      prob = (1 - mix) * learnedProb + mix * vote.score;
    }
    if (extracted.unreadable) prob = Math.min(prob, 0.35);

    var status = "unsure";
    if (extracted.unreadable) status = "unreadable";
    else if (count < 4) status = "learning";
    else if (prob >= 0.62) status = "yes";
    else if (prob <= 0.38) status = "no";

    var pos = 0;
    var neg = 0;
    for (var i = 0; i < model.examples.length; i++) {
      if (model.examples[i].label) pos += 1;
      else neg += 1;
    }

    return {
      prob: prob,
      priorProb: priorProb,
      learnedProb: learnedProb,
      status: status,
      cues: extracted.cues || [],
      path: extracted.path || [],
      unreadable: !!extracted.unreadable,
      neighbors: vote.neighbors.map(function (row) {
        return {
          id: row.example.id,
          label: row.example.label,
          note: row.example.note,
          name: row.example.name,
          thumb: row.example.thumb,
          sim: row.sim
        };
      }),
      counts: { total: count, pos: pos, neg: neg }
    };
  }

  function exportState(model, rules) {
    return {
      version: 1,
      rules: rules || DEFAULT_RULES,
      examples: model.examples.map(function (ex) {
        return {
          id: ex.id,
          label: ex.label,
          note: ex.note,
          name: ex.name,
          createdAt: ex.createdAt,
          source: ex.source,
          vector: ex.vector,
          thumb: ex.thumb || ""
        };
      })
    };
  }

  function importExamples(model, payload) {
    var list = Array.isArray(payload) ? payload : (payload && payload.examples) || [];
    var seen = {};
    model.examples.forEach(function (ex) { seen[ex.id] = true; });
    var added = 0;
    list.forEach(function (ex) {
      if (!ex || !ex.vector || ex.vector.length !== FEATURE_DIM) return;
      if (ex.id && seen[ex.id]) return;
      var id = ex.id || ("ex_" + Date.now().toString(36) + "_" + Math.random().toString(36).slice(2, 8));
      seen[id] = true;
      model.examples.push({
        id: id,
        label: ex.label ? 1 : 0,
        note: String(ex.note || "").slice(0, 280),
        name: String(ex.name || "").slice(0, 120),
        createdAt: ex.createdAt || new Date().toISOString(),
        source: ex.source || "import",
        vector: roundVector(ex.vector),
        thumb: ex.thumb || ""
      });
      added += 1;
    });
    refit(model);
    return added;
  }

  return {
    FEATURE_DIM: FEATURE_DIM,
    INDEX: I,
    DEFAULT_RULES: DEFAULT_RULES,
    extractChartFeatures: extractChartFeatures,
    createModel: createModel,
    refit: refit,
    addExample: addExample,
    removeExample: removeExample,
    predict: predict,
    exportState: exportState,
    importExamples: importExamples,
    shapeScoreOf: shapeScoreOf,
    downscale: downscale
  };
});
