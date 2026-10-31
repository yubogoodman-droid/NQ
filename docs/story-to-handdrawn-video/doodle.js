(function (root) {
  "use strict";

  const STYLES = {
    diary: {
      id: "diary",
      name: "彩铅日记",
      ink: "#2a241c",
      paper: "#fffefb",
      caption: "#2a241c",
      palette: ["#6b8ca8", "#b85c4a", "#c4a574", "#e0c45c", "#8a8074", "#d4c4a8", "#7a9e8a"],
      stroke: 2.35,
      crayon: true,
      wash: false,
    },
    crayon: {
      id: "crayon",
      name: "儿童蜡笔",
      ink: "#3a2a18",
      paper: "#fffaf0",
      caption: "#3a2a18",
      palette: ["#e07a4a", "#f0c14b", "#6bb0d6", "#7bc67a", "#d46aa0", "#c4a574"],
      stroke: 3.4,
      crayon: true,
      wash: false,
    },
    line: {
      id: "line",
      name: "极简线条",
      ink: "#1a1a1a",
      paper: "#ffffff",
      caption: "#1a1a1a",
      palette: ["#d0d0d0", "#9aa3ad", "#c8b8a4"],
      stroke: 1.7,
      crayon: false,
      wash: false,
    },
    ink: {
      id: "ink",
      name: "水墨",
      ink: "#161616",
      paper: "#f7f4ee",
      caption: "#161616",
      palette: ["#2a2a2a", "#6a6a6a", "#9a9590", "#c4c0b8"],
      stroke: 2.1,
      crayon: false,
      wash: true,
    },
  };

  const MOTIFS = [
    { tag: "cat", re: /猫|喵|kitten|cat/i },
    { tag: "bird", re: /鸟|雀|鸟鸦|bird|sparrow/i },
    { tag: "dog", re: /狗|汪|dog/i },
    { tag: "child", re: /小孩|孩子|小朋友|儿童|男孩|女孩|child|boy|girl/i },
    { tag: "elder", re: /爷爷|奶奶|外婆|外公|老人|grandmother|grandfather/i },
    { tag: "rain", re: /雨|rain/i },
    { tag: "sun", re: /阳光|太阳|晴|sun/i },
    { tag: "night", re: /夜|晚|月亮|星星|moon|star|night/i },
    { tag: "window", re: /窗|window/i },
    { tag: "tree", re: /树|枝|林|tree|branch/i },
    { tag: "flower", re: /花|flower/i },
    { tag: "house", re: /家|屋|房|house|home/i },
    { tag: "sea", re: /海|河|湖|sea|river|lake/i },
    { tag: "tea", re: /茶|汤|碗|杯|tea|soup|cup/i },
    { tag: "book", re: /书|信|book|letter/i },
    { tag: "sleep", re: /睡|梦|sleep|dream/i },
    { tag: "walk", re: /走|散步|walk/i },
    { tag: "heart", re: /爱|心|喜欢|love|heart/i },
    { tag: "snow", re: /雪|snow/i },
    { tag: "cloud", re: /云|cloud/i },
    { tag: "umbrella", re: /伞|umbrella/i },
    { tag: "hello", re: /打招呼|招手|hello|hi/i },
  ];

  function mulberry32(a) {
    return function () {
      let t = (a += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function hashStr(s) {
    let h = 2166136261;
    for (let i = 0; i < s.length; i++) h = Math.imul(h ^ s.charCodeAt(i), 16777619);
    return h >>> 0;
  }

  function parseMotifs(text) {
    const tags = [];
    for (const m of MOTIFS) if (m.re.test(text)) tags.push(m.tag);
    return tags;
  }

  function wobbleCircle(ctx, x, y, r, rng, amp, steps) {
    steps = steps || 18;
    ctx.beginPath();
    for (let i = 0; i <= steps; i++) {
      const a = (i / steps) * Math.PI * 2;
      const rr = r + (rng() - 0.5) * amp;
      const px = x + Math.cos(a) * rr;
      const py = y + Math.sin(a) * rr * 0.96;
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  function wobbleLine(ctx, x1, y1, x2, y2, rng, amp, segs) {
    segs = segs || 7;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    for (let i = 1; i <= segs; i++) {
      const t = i / segs;
      const x = x1 + (x2 - x1) * t + (rng() - 0.5) * amp;
      const y = y1 + (y2 - y1) * t + (rng() - 0.5) * amp;
      ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  function crayonFill(ctx, pathFn, bounds, color, rng, n) {
    n = n == null ? 70 : n;
    ctx.save();
    pathFn();
    ctx.clip();
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.2;
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.globalAlpha = 0.42;
    ctx.lineWidth = 1.55;
    ctx.lineCap = "round";
    const { x, y, w, h } = bounds;
    for (let i = 0; i < n; i++) {
      const px = x + rng() * w;
      const py = y + rng() * h;
      const ang = rng() * Math.PI;
      const len = 5 + rng() * 10;
      ctx.beginPath();
      ctx.moveTo(px, py);
      ctx.lineTo(px + Math.cos(ang) * len, py + Math.sin(ang) * len);
      ctx.stroke();
    }
    ctx.restore();
  }

  function inkWash(ctx, pathFn, color, rng) {
    ctx.save();
    ctx.globalAlpha = 0.12 + rng() * 0.1;
    ctx.fillStyle = color;
    pathFn();
    ctx.fill();
    ctx.restore();
  }

  function strokeInk(ctx, style) {
    ctx.strokeStyle = style.ink;
    ctx.lineWidth = style.stroke;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
  }

  function fillShape(ctx, style, pathFn, bounds, color, rng) {
    if (style.wash) inkWash(ctx, pathFn, color, rng);
    else if (style.crayon) crayonFill(ctx, pathFn, bounds, color, rng);
    else {
      ctx.save();
      ctx.globalAlpha = 0.28;
      ctx.fillStyle = color;
      pathFn();
      ctx.fill();
      ctx.restore();
    }
    strokeInk(ctx, style);
    pathFn();
    ctx.stroke();
  }

  function drawCat(ctx, x, y, s, style, rng, opts) {
    opts = opts || {};
    const fur = style.palette[1] || "#b85c4a";
    const cream = style.palette[2] || "#c4a574";
    const sleeping = opts.sleep;
    const paw = opts.paw;

    const body = () => wobbleCircle(ctx, x, y + 18 * s, 28 * s, rng, 3.2 * s, 16);
    fillShape(ctx, style, body, { x: x - 32 * s, y: y - 8 * s, w: 64 * s, h: 56 * s }, cream, rng);

    const head = () => wobbleCircle(ctx, x, y - 22 * s, 26 * s, rng, 2.6 * s, 16);
    fillShape(ctx, style, head, { x: x - 30 * s, y: y - 50 * s, w: 60 * s, h: 56 * s }, cream, rng);

    ctx.save();
    strokeInk(ctx, style);
    ctx.fillStyle = cream;
    ctx.beginPath();
    ctx.moveTo(x - 18 * s, y - 38 * s);
    ctx.lineTo(x - 28 * s + (rng() - 0.5) * 2, y - 58 * s);
    ctx.lineTo(x - 6 * s, y - 42 * s);
    ctx.closePath();
    ctx.globalAlpha = 0.35;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x + 18 * s, y - 38 * s);
    ctx.lineTo(x + 28 * s + (rng() - 0.5) * 2, y - 58 * s);
    ctx.lineTo(x + 6 * s, y - 42 * s);
    ctx.closePath();
    ctx.globalAlpha = 0.35;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = fur;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 2.2 * s;
    for (let i = 0; i < 3; i++) {
      wobbleLine(ctx, x - 10 * s, y - 8 * s + i * 7 * s, x + 12 * s, y - 4 * s + i * 7 * s, rng, 1.5, 4);
    }
    ctx.restore();

    ctx.save();
    ctx.fillStyle = style.ink;
    if (sleeping) {
      ctx.lineWidth = 1.8 * s;
      ctx.strokeStyle = style.ink;
      ctx.beginPath();
      ctx.arc(x - 9 * s, y - 24 * s, 4 * s, Math.PI, 0);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(x + 9 * s, y - 24 * s, 4 * s, Math.PI, 0);
      ctx.stroke();
    } else {
      ctx.beginPath();
      ctx.arc(x - 8 * s, y - 24 * s, 3.1 * s, 0, Math.PI * 2);
      ctx.arc(x + 8 * s, y - 24 * s, 3.1 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.fillStyle = fur;
    ctx.beginPath();
    ctx.moveTo(x, y - 18 * s);
    ctx.lineTo(x - 3 * s, y - 13 * s);
    ctx.lineTo(x + 3 * s, y - 13 * s);
    ctx.fill();
    ctx.restore();

    ctx.save();
    strokeInk(ctx, style);
    ctx.lineWidth = 1.2 * s;
    wobbleLine(ctx, x - 4 * s, y - 16 * s, x - 22 * s, y - 12 * s, rng, 0.8, 3);
    wobbleLine(ctx, x - 4 * s, y - 14 * s, x - 22 * s, y - 18 * s, rng, 0.8, 3);
    wobbleLine(ctx, x + 4 * s, y - 16 * s, x + 22 * s, y - 12 * s, rng, 0.8, 3);
    wobbleLine(ctx, x + 4 * s, y - 14 * s, x + 22 * s, y - 18 * s, rng, 0.8, 3);
    ctx.restore();

    ctx.save();
    strokeInk(ctx, style);
    ctx.beginPath();
    ctx.moveTo(x - 24 * s, y + 16 * s);
    ctx.quadraticCurveTo(x - 48 * s, y + 8 * s, x - 40 * s, y + 32 * s);
    ctx.stroke();
    ctx.restore();

    if (paw) {
      ctx.save();
      strokeInk(ctx, style);
      ctx.lineWidth = 2.4 * s;
      wobbleLine(ctx, x + 18 * s, y + 4 * s, x + 38 * s, y - 10 * s, rng, 1.2, 4);
      ctx.restore();
    }
  }

  function drawBird(ctx, x, y, s, style, rng, opts) {
    opts = opts || {};
    const blue = style.palette[0] || "#6b8ca8";
    const beak = style.palette[3] || "#e0c45c";
    const body = () => wobbleCircle(ctx, x, y, 22 * s, rng, 2.4 * s, 16);
    fillShape(ctx, style, body, { x: x - 26 * s, y: y - 24 * s, w: 52 * s, h: 48 * s }, blue, rng);

    ctx.save();
    strokeInk(ctx, style);
    ctx.beginPath();
    ctx.moveTo(x + 4 * s, y + 2 * s);
    ctx.quadraticCurveTo(x + 22 * s, y - 6 * s, x + 16 * s, y + 16 * s);
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = beak;
    ctx.beginPath();
    ctx.moveTo(x + 20 * s, y - 2 * s);
    ctx.lineTo(x + 32 * s, y + 2 * s);
    ctx.lineTo(x + 20 * s, y + 6 * s);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = style.ink;
    ctx.lineWidth = 1.2;
    ctx.stroke();
    ctx.restore();

    ctx.save();
    ctx.fillStyle = style.ink;
    if (opts.tilt) {
      ctx.beginPath();
      ctx.arc(x + 2 * s, y - 8 * s, 2.7 * s, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.beginPath();
      ctx.arc(x - 4 * s, y - 6 * s, 2.7 * s, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    if (opts.shake) {
      ctx.save();
      ctx.strokeStyle = style.palette[4] || "#8a8074";
      ctx.globalAlpha = 0.7;
      ctx.lineWidth = 1.4;
      wobbleLine(ctx, x - 8 * s, y - 26 * s, x - 2 * s, y - 38 * s, rng, 1, 3);
      wobbleLine(ctx, x + 6 * s, y - 24 * s, x + 14 * s, y - 36 * s, rng, 1, 3);
      ctx.restore();
    }
  }

  function drawChild(ctx, x, y, s, style, rng) {
    const skin = style.palette[2] || "#c4a574";
    const shirt = style.palette[0] || "#6b8ca8";
    const head = () => wobbleCircle(ctx, x, y - 28 * s, 22 * s, rng, 2.2 * s, 14);
    fillShape(ctx, style, head, { x: x - 26 * s, y: y - 52 * s, w: 52 * s, h: 48 * s }, skin, rng);
    const body = () => wobbleCircle(ctx, x, y + 10 * s, 20 * s, rng, 2.4 * s, 12);
    fillShape(ctx, style, body, { x: x - 24 * s, y: y - 8 * s, w: 48 * s, h: 44 * s }, shirt, rng);
    ctx.save();
    ctx.fillStyle = style.ink;
    ctx.beginPath();
    ctx.arc(x - 7 * s, y - 30 * s, 2.5 * s, 0, Math.PI * 2);
    ctx.arc(x + 7 * s, y - 30 * s, 2.5 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = style.ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x, y - 22 * s, 6 * s, 0.15 * Math.PI, 0.85 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  function drawElder(ctx, x, y, s, style, rng) {
    const skin = style.palette[5] || "#d4c4a8";
    const robe = style.palette[4] || "#8a8074";
    const head = () => wobbleCircle(ctx, x, y - 26 * s, 20 * s, rng, 2 * s, 14);
    fillShape(ctx, style, head, { x: x - 24 * s, y: y - 48 * s, w: 48 * s, h: 44 * s }, skin, rng);
    ctx.save();
    strokeInk(ctx, style);
    ctx.beginPath();
    ctx.moveTo(x - 16 * s, y + 4 * s);
    ctx.quadraticCurveTo(x, y + 44 * s, x + 18 * s, y + 6 * s);
    ctx.stroke();
    ctx.restore();
    const body = () => {
      ctx.beginPath();
      ctx.moveTo(x - 18 * s, y - 6 * s);
      ctx.lineTo(x - 26 * s, y + 40 * s);
      ctx.lineTo(x + 26 * s, y + 40 * s);
      ctx.lineTo(x + 18 * s, y - 6 * s);
      ctx.closePath();
    };
    fillShape(ctx, style, body, { x: x - 28 * s, y: y - 8 * s, w: 56 * s, h: 50 * s }, robe, rng);
    ctx.save();
    ctx.fillStyle = style.ink;
    ctx.beginPath();
    ctx.arc(x - 6 * s, y - 28 * s, 2.2 * s, 0, Math.PI * 2);
    ctx.arc(x + 6 * s, y - 28 * s, 2.2 * s, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  function drawWindow(ctx, x, y, w, h, style, rng) {
    const frame = style.palette[0] || "#6b8ca8";
    ctx.save();
    ctx.strokeStyle = frame;
    ctx.lineWidth = style.stroke + 1.2;
    ctx.lineJoin = "round";
    wobbleLine(ctx, x, y, x, y + h, rng, 1.6, 5);
    wobbleLine(ctx, x + w, y, x + w, y + h, rng, 1.6, 5);
    wobbleLine(ctx, x, y + h, x + w, y + h, rng, 1.4, 5);
    wobbleLine(ctx, x + w * 0.5, y + 8, x + w * 0.5, y + h - 4, rng, 1.2, 4);
    ctx.restore();
  }

  function drawRain(ctx, x, y, w, h, style, rng, n) {
    ctx.save();
    ctx.strokeStyle = style.palette[4] || "#8a8074";
    ctx.globalAlpha = 0.55;
    ctx.lineWidth = 1.35;
    n = n || 18;
    for (let i = 0; i < n; i++) {
      const px = x + rng() * w;
      const py = y + rng() * h;
      wobbleLine(ctx, px, py, px + 6, py + 18, rng, 0.6, 2);
    }
    ctx.restore();
  }

  function drawSun(ctx, x, y, r, style, rng) {
    const gold = style.palette[3] || "#e0c45c";
    const disk = () => wobbleCircle(ctx, x, y, r, rng, 2, 14);
    fillShape(ctx, style, disk, { x: x - r - 4, y: y - r - 4, w: r * 2 + 8, h: r * 2 + 8 }, gold, rng, 40);
    ctx.save();
    ctx.strokeStyle = gold;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1.6;
    for (let i = 0; i < 8; i++) {
      const a = (i / 8) * Math.PI * 2;
      wobbleLine(
        ctx,
        x + Math.cos(a) * (r + 6),
        y + Math.sin(a) * (r + 6),
        x + Math.cos(a) * (r + 16),
        y + Math.sin(a) * (r + 16),
        rng,
        1,
        2
      );
    }
    ctx.restore();
  }

  function drawMoon(ctx, x, y, r, style, rng) {
    const cream = style.palette[5] || "#d4c4a8";
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r, 0, Math.PI * 2);
    ctx.arc(x + r * 0.38, y - r * 0.1, r * 0.82, 0, Math.PI * 2);
    ctx.clip("evenodd");
    ctx.fillStyle = cream;
    ctx.globalAlpha = 0.35;
    ctx.fillRect(x - r, y - r, r * 2, r * 2);
    ctx.restore();
    ctx.save();
    strokeInk(ctx, style);
    ctx.beginPath();
    ctx.arc(x, y, r, 0.35 * Math.PI, 1.55 * Math.PI);
    ctx.stroke();
    ctx.restore();
  }

  function drawStars(ctx, style, rng, n) {
    ctx.save();
    ctx.fillStyle = style.palette[3] || "#e0c45c";
    for (let i = 0; i < (n || 7); i++) {
      const x = 80 + rng() * 560;
      const y = 250 + rng() * 180;
      ctx.globalAlpha = 0.45 + rng() * 0.4;
      ctx.beginPath();
      ctx.arc(x, y, 1.6 + rng() * 1.8, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  function drawTree(ctx, x, y, s, style, rng) {
    const brown = style.palette[2] || "#c4a574";
    const leaf = style.palette[6] || style.palette[0];
    ctx.save();
    ctx.strokeStyle = brown;
    ctx.lineWidth = 4 * s;
    wobbleLine(ctx, x, y, x, y + 70 * s, rng, 2, 4);
    ctx.restore();
    const crown = () => wobbleCircle(ctx, x, y, 28 * s, rng, 4 * s, 12);
    fillShape(ctx, style, crown, { x: x - 32 * s, y: y - 32 * s, w: 64 * s, h: 64 * s }, leaf, rng);
  }

  function drawBranch(ctx, x, y, w, style, rng) {
    ctx.save();
    ctx.strokeStyle = style.palette[2] || "#c4a574";
    ctx.lineWidth = 3.2;
    wobbleLine(ctx, x, y, x + w, y + 8, rng, 2, 6);
    ctx.lineWidth = 1.6;
    wobbleLine(ctx, x + w * 0.4, y + 2, x + w * 0.55, y - 16, rng, 1.2, 3);
    ctx.restore();
  }

  function drawHouse(ctx, x, y, s, style, rng) {
    const wall = style.palette[5] || "#d4c4a8";
    const roof = style.palette[1] || "#b85c4a";
    const body = () => {
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + 70 * s, y);
      ctx.lineTo(x + 70 * s, y + 48 * s);
      ctx.lineTo(x, y + 48 * s);
      ctx.closePath();
    };
    fillShape(ctx, style, body, { x, y, w: 70 * s, h: 48 * s }, wall, rng, 40);
    ctx.save();
    ctx.strokeStyle = style.ink;
    ctx.lineWidth = style.stroke;
    ctx.beginPath();
    ctx.moveTo(x - 8 * s, y + 4 * s);
    ctx.lineTo(x + 35 * s, y - 28 * s);
    ctx.lineTo(x + 78 * s, y + 4 * s);
    ctx.stroke();
    ctx.restore();
    const roofFill = () => {
      ctx.beginPath();
      ctx.moveTo(x - 8 * s, y + 4 * s);
      ctx.lineTo(x + 35 * s, y - 28 * s);
      ctx.lineTo(x + 78 * s, y + 4 * s);
      ctx.closePath();
    };
    if (style.crayon) crayonFill(ctx, roofFill, { x: x - 8 * s, y: y - 28 * s, w: 86 * s, h: 36 * s }, roof, rng, 30);
  }

  function drawFlower(ctx, x, y, s, style, rng) {
    const petal = style.palette[1] || "#b85c4a";
    const center = style.palette[3] || "#e0c45c";
    ctx.save();
    ctx.strokeStyle = style.palette[6] || "#7a9e8a";
    ctx.lineWidth = 1.8;
    wobbleLine(ctx, x, y, x, y + 28 * s, rng, 1, 3);
    ctx.restore();
    for (let i = 0; i < 5; i++) {
      const a = (i / 5) * Math.PI * 2;
      const px = x + Math.cos(a) * 8 * s;
      const py = y + Math.sin(a) * 8 * s;
      const p = () => wobbleCircle(ctx, px, py, 6 * s, rng, 1.2, 8);
      fillShape(ctx, style, p, { x: px - 8 * s, y: py - 8 * s, w: 16 * s, h: 16 * s }, petal, rng, 12);
    }
    const c = () => wobbleCircle(ctx, x, y, 4.5 * s, rng, 0.8, 8);
    fillShape(ctx, style, c, { x: x - 6 * s, y: y - 6 * s, w: 12 * s, h: 12 * s }, center, rng, 8);
  }

  function drawTea(ctx, x, y, s, style, rng) {
    const cup = style.palette[5] || "#d4c4a8";
    const body = () => {
      ctx.beginPath();
      ctx.moveTo(x - 22 * s, y);
      ctx.lineTo(x - 16 * s, y + 28 * s);
      ctx.lineTo(x + 16 * s, y + 28 * s);
      ctx.lineTo(x + 22 * s, y);
      ctx.closePath();
    };
    fillShape(ctx, style, body, { x: x - 24 * s, y, w: 48 * s, h: 30 * s }, cup, rng, 30);
    ctx.save();
    strokeInk(ctx, style);
    ctx.beginPath();
    ctx.arc(x + 24 * s, y + 12 * s, 8 * s, -0.4 * Math.PI, 0.4 * Math.PI);
    ctx.stroke();
    ctx.globalAlpha = 0.4;
    wobbleLine(ctx, x - 6 * s, y - 8 * s, x - 4 * s, y - 22 * s, rng, 1, 3);
    wobbleLine(ctx, x + 4 * s, y - 6 * s, x + 8 * s, y - 20 * s, rng, 1, 3);
    ctx.restore();
  }

  function drawBook(ctx, x, y, s, style, rng) {
    const cover = style.palette[1] || "#b85c4a";
    const body = () => {
      ctx.beginPath();
      ctx.rect(x, y, 48 * s, 36 * s);
    };
    fillShape(ctx, style, body, { x, y, w: 48 * s, h: 36 * s }, cover, rng, 24);
    ctx.save();
    strokeInk(ctx, style);
    wobbleLine(ctx, x + 24 * s, y, x + 24 * s, y + 36 * s, rng, 0.8, 3);
    ctx.restore();
  }

  function drawSea(ctx, y, style, rng) {
    ctx.save();
    ctx.strokeStyle = style.palette[0] || "#6b8ca8";
    ctx.globalAlpha = 0.65;
    ctx.lineWidth = 1.8;
    for (let i = 0; i < 4; i++) {
      ctx.beginPath();
      let px = 40;
      ctx.moveTo(px, y + i * 14);
      while (px < 680) {
        px += 28;
        ctx.quadraticCurveTo(px - 10, y + i * 14 + (rng() - 0.5) * 10, px, y + i * 14);
      }
      ctx.stroke();
    }
    ctx.restore();
  }

  function drawCloud(ctx, x, y, s, style, rng) {
    const c = style.palette[5] || "#d4c4a8";
    const a = () => wobbleCircle(ctx, x, y, 16 * s, rng, 2, 10);
    const b = () => wobbleCircle(ctx, x + 18 * s, y + 2 * s, 14 * s, rng, 2, 10);
    fillShape(ctx, style, a, { x: x - 18 * s, y: y - 18 * s, w: 36 * s, h: 36 * s }, c, rng, 16);
    fillShape(ctx, style, b, { x: x, y: y - 14 * s, w: 32 * s, h: 32 * s }, c, rng, 12);
  }

  function drawUmbrella(ctx, x, y, s, style, rng) {
    const canopy = style.palette[1] || "#b85c4a";
    const cap = () => {
      ctx.beginPath();
      ctx.moveTo(x - 34 * s, y);
      ctx.quadraticCurveTo(x, y - 28 * s, x + 34 * s, y);
      ctx.closePath();
    };
    fillShape(ctx, style, cap, { x: x - 36 * s, y: y - 28 * s, w: 72 * s, h: 32 * s }, canopy, rng, 28);
    ctx.save();
    strokeInk(ctx, style);
    wobbleLine(ctx, x, y, x, y + 38 * s, rng, 1, 3);
    ctx.restore();
  }

  function drawHeart(ctx, x, y, s, style, rng) {
    const c = style.palette[1] || "#b85c4a";
    const h = () => {
      ctx.beginPath();
      ctx.moveTo(x, y + 10 * s);
      ctx.bezierCurveTo(x - 22 * s, y - 8 * s, x - 10 * s, y - 22 * s, x, y - 8 * s);
      ctx.bezierCurveTo(x + 10 * s, y - 22 * s, x + 22 * s, y - 8 * s, x, y + 10 * s);
    };
    fillShape(ctx, style, h, { x: x - 22 * s, y: y - 22 * s, w: 44 * s, h: 36 * s }, c, rng, 20);
  }

  function drawSill(ctx, y, style, rng) {
    ctx.save();
    ctx.strokeStyle = style.palette[2] || "#c4a574";
    ctx.lineWidth = 3;
    wobbleLine(ctx, 70, y, 650, y, rng, 2, 8);
    ctx.globalAlpha = 0.25;
    ctx.fillStyle = style.palette[5] || "#d4c4a8";
    ctx.fillRect(70, y, 580, 14);
    ctx.restore();
  }

  function composeScene(ctx, W, H, sentence, styleId, seedText) {
    const style = STYLES[styleId] || STYLES.diary;
    const rng = mulberry32(hashStr((seedText || sentence) + "|" + style.id));
    const tags = parseMotifs(sentence);

    ctx.save();
    ctx.fillStyle = style.paper;
    ctx.fillRect(0, 0, W, H);
    ctx.restore();

    const has = (t) => tags.indexOf(t) !== -1;
    const top = H * 0.24;

    if (has("night")) drawStars(ctx, style, rng, 9);
    if (has("sun") && !has("night")) drawSun(ctx, W - 120, top + 70, 28, style, rng);
    if (has("night")) drawMoon(ctx, W - 130, top + 70, 26, style, rng);
    if (has("cloud") || has("walk")) drawCloud(ctx, 140, top + 60, 1.1, style, rng);

    if (has("window")) {
      drawWindow(ctx, 210, top + 40, 340, 430, style, rng);
      drawSill(ctx, top + 470, style, rng);
    }

    if (has("rain")) {
      drawRain(ctx, has("window") ? 230 : 80, top + 50, has("window") ? 300 : 560, 360, style, rng, has("window") ? 16 : 22);
    }
    if (has("snow")) {
      ctx.save();
      ctx.fillStyle = style.palette[5] || "#d4c4a8";
      for (let i = 0; i < 22; i++) {
        ctx.globalAlpha = 0.5;
        ctx.beginPath();
        ctx.arc(80 + rng() * 560, top + 40 + rng() * 420, 2 + rng() * 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    }

    if (has("house") && !has("window")) drawHouse(ctx, 430, top + 280, 1.15, style, rng);
    if (has("tree") && !has("bird")) drawTree(ctx, 160, top + 250, 1.15, style, rng);
    if (has("sea")) drawSea(ctx, H - 220, style, rng);
    if (has("flower")) {
      drawFlower(ctx, 150, H - 260, 1.1, style, rng);
      drawFlower(ctx, 200, H - 240, 0.85, style, rng);
    }

    if (has("bird")) {
      if (has("tree") || /枝/.test(sentence)) drawBranch(ctx, 180, top + 280, 280, style, rng);
      drawBird(ctx, has("cat") ? 500 : 360, has("cat") ? top + 250 : top + 260, 1.35, style, rng, {
        shake: /抖/.test(sentence),
        tilt: /歪/.test(sentence),
      });
    }

    if (has("cat")) {
      const cx = has("window") ? 230 : has("bird") ? 250 : 360;
      drawCat(ctx, cx, top + 340, 1.45, style, rng, {
        sleep: has("sleep") || /睡/.test(sentence),
        paw: has("hello") || /爪子|招/.test(sentence),
      });
    }

    if (has("dog") && !has("cat")) {
      drawCat(ctx, 300, top + 360, 1.3, style, rng, {});
    }

    if (has("elder")) drawElder(ctx, has("child") ? 250 : 360, top + 340, 1.25, style, rng);
    if (has("child")) drawChild(ctx, has("elder") ? 470 : 360, top + 360, 1.25, style, rng);

    if (!has("cat") && !has("bird") && !has("child") && !has("elder") && !has("dog")) {
      drawChild(ctx, 340, top + 360, 1.3, style, rng);
    }

    if (has("tea")) drawTea(ctx, has("child") || has("elder") ? 520 : 360, top + 430, 1.2, style, rng);
    if (has("book")) drawBook(ctx, 160, top + 430, 1.15, style, rng);
    if (has("umbrella") || (has("rain") && (has("child") || has("walk")))) {
      if (!has("window")) drawUmbrella(ctx, 500, top + 220, 1.1, style, rng);
    }
    if (has("heart")) drawHeart(ctx, W / 2, top + 80, 1.1, style, rng);
    if (has("sleep") && !has("cat")) {
      ctx.save();
      ctx.font = "28px 'Ma Shan Zheng', cursive";
      ctx.fillStyle = style.caption;
      ctx.globalAlpha = 0.45;
      ctx.fillText("z z", 480, top + 160);
      ctx.restore();
    }

    if (has("walk") && !has("window")) {
      ctx.save();
      ctx.strokeStyle = style.palette[4] || "#8a8074";
      ctx.globalAlpha = 0.4;
      wobbleLine(ctx, 80, H - 180, 640, H - 160, rng, 2, 8);
      ctx.restore();
    }

    return { tags, style: style.id };
  }

  function renderPage(colorCanvas, bwCanvas, sentence, styleId, seedText) {
    const ctx = colorCanvas.getContext("2d");
    composeScene(ctx, colorCanvas.width, colorCanvas.height, sentence, styleId, seedText);
    const bw = bwCanvas.getContext("2d");
    bw.filter = "grayscale(1) contrast(1.25) brightness(1.05)";
    bw.drawImage(colorCanvas, 0, 0);
    bw.filter = "none";
  }

  root.HandDoodle = {
    STYLES,
    parseMotifs,
    composeScene,
    renderPage,
  };
})(typeof globalThis !== "undefined" ? globalThis : this);
