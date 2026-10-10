(function (root) {
  "use strict";

  const W = 720;
  const H = 960;
  const PAGE_SEC = 4.8;
  const FLIP_SEC = 0.7;

  function easeOutCubic(t) {
    return 1 - Math.pow(1 - t, 3);
  }
  function clamp(v, a, b) {
    return Math.max(a, Math.min(b, v));
  }
  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("image failed: " + src));
      img.src = src;
    });
  }

  function wrapText(ctx, text, maxWidth) {
    const chars = Array.from(text || "");
    const lines = [];
    let line = "";
    for (const ch of chars) {
      const trial = line + ch;
      if (ctx.measureText(trial).width > maxWidth && line) {
        lines.push(line);
        line = ch;
      } else line = trial;
    }
    if (line) lines.push(line);
    return lines.slice(0, 3);
  }

  function makeOffscreen() {
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    return c;
  }

  function desaturateTo(srcCanvas, destCanvas) {
    const dctx = destCanvas.getContext("2d");
    dctx.filter = "grayscale(1) contrast(1.35) brightness(1.05)";
    dctx.drawImage(srcCanvas, 0, 0);
    dctx.filter = "none";
  }

  function drawCaption(ctx, page, progress) {
    const isCover = page.kind === "cover";
    ctx.save();
    ctx.fillStyle = "#2a241c";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    const font = isCover
      ? "700 54px 'Ma Shan Zheng', 'Liu Jian Mao Cao', cursive"
      : "40px 'Ma Shan Zheng', 'Liu Jian Mao Cao', cursive";
    ctx.font = font;

    const wipe = clamp(progress, 0, 1);
    ctx.beginPath();
    ctx.rect(0, 0, W * wipe, H * 0.28);
    ctx.clip();

    const lines = wrapText(ctx, page.caption || "", isCover ? W - 96 : W - 88);
    const y0 = isCover ? 92 : 78;
    lines.forEach((ln, i) => {
      ctx.fillText(ln, W / 2, y0 + i * (isCover ? 58 : 46));
    });
    if (isCover && page.sub) {
      ctx.font = "22px 'Noto Sans TC', sans-serif";
      ctx.fillStyle = "#8a7b68";
      ctx.fillText(page.sub, W / 2, y0 + lines.length * 58 + 8);
    }
    ctx.restore();

    ctx.save();
    ctx.font = "16px 'Noto Sans TC', sans-serif";
    ctx.fillStyle = "rgba(42,36,28,0.35)";
    ctx.textAlign = "right";
    ctx.fillText(page.indexLabel || "", W - 36, H - 28);
    ctx.restore();
  }

  function wipeImage(ctx, img, progress) {
    const p = clamp(progress, 0, 1);
    if (p <= 0) return;
    ctx.save();
    ctx.beginPath();
    ctx.rect(0, 0, W * p, H);
    ctx.clip();
    ctx.drawImage(img, 0, 0, W, H);
    ctx.restore();
  }

  function drawCurl(ctx, from, to, p) {
    ctx.drawImage(to, 0, 0, W, H);
    if (p <= 0.001) {
      ctx.drawImage(from, 0, 0, W, H);
      return;
    }
    const reach = easeOutCubic(clamp(p, 0, 1)) * 1.02;
    const x0 = W * (1 - reach);
    const y0 = H * (1 - reach);

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(0, 0);
    ctx.lineTo(W, 0);
    ctx.lineTo(W, Math.max(0, y0));
    ctx.lineTo(Math.max(0, x0), H);
    ctx.lineTo(0, H);
    ctx.closePath();
    ctx.clip();
    ctx.drawImage(from, 0, 0, W, H);
    ctx.restore();

    const tipX = W - (W - x0) * 0.52;
    const tipY = H - (H - y0) * 0.52;
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(W, y0);
    ctx.lineTo(x0, H);
    ctx.lineTo(tipX, tipY);
    ctx.closePath();
    ctx.clip();
    ctx.globalAlpha = 0.62;
    ctx.filter = "brightness(1.18) saturate(0.25)";
    ctx.drawImage(from, 0, 0, W, H);
    ctx.filter = "none";
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(40,30,20,0.28)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(W, y0);
    ctx.lineTo(x0, H);
    ctx.stroke();
    const shadow = ctx.createLinearGradient(W, y0, x0, H);
    shadow.addColorStop(0, "rgba(0,0,0,0.16)");
    shadow.addColorStop(1, "rgba(0,0,0,0)");
    ctx.strokeStyle = shadow;
    ctx.lineWidth = 14;
    ctx.beginPath();
    ctx.moveTo(W, y0);
    ctx.lineTo(x0, H);
    ctx.stroke();
    ctx.restore();
  }

  function layeredProgress(local) {
    const t = local;
    return {
      text: clamp(t / 1.05, 0, 1),
      bw: clamp((t - 0.95) / 1.2, 0, 1),
      color: clamp((t - 2.1) / 1.2, 0, 1),
    };
  }

  class StoryPlayer {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext("2d");
      canvas.width = W;
      canvas.height = H;
      this.pages = [];
      this.transition = "cut";
      this.pageSec = PAGE_SEC;
      this.playing = false;
      this.t0 = 0;
      this.elapsed = 0;
      this.raf = 0;
      this.onFrame = null;
      this.onEnded = null;
      this._fromBuf = makeOffscreen();
      this._toBuf = makeOffscreen();
    }

    get totalSec() {
      const n = this.pages.length;
      if (!n) return 0;
      const flip = this.transition === "page-flip" ? FLIP_SEC : 0;
      return n * this.pageSec + Math.max(0, n - 1) * flip;
    }

    async loadAiPages(story) {
      const pages = [];
      for (let i = 0; i < story.pages.length; i++) {
        const p = story.pages[i];
        const color = await loadImage(p.color);
        const bw = await loadImage(p.bw);
        pages.push({
          kind: p.kind,
          caption: p.caption,
          sub: p.sub,
          color,
          bw,
          indexLabel: String(i + 1).padStart(2, "0"),
        });
      }
      this.pages = pages;
      this.drawAt(0);
    }

    loadDoodlePages(title, beats, styleId) {
      const pages = [];
      const coverC = makeOffscreen();
      const coverB = makeOffscreen();
      const coverPrompt = title + " " + (beats[0] || "");
      root.HandDoodle.renderPage(coverC, coverB, coverPrompt, styleId, title + "-cover");
      pages.push({
        kind: "cover",
        caption: title,
        sub: (root.HandDoodle.STYLES[styleId] || {}).name || "手绘",
        color: coverC,
        bw: coverB,
        indexLabel: "01",
      });
      beats.forEach((beat, i) => {
        const c = makeOffscreen();
        const b = makeOffscreen();
        root.HandDoodle.renderPage(c, b, beat, styleId, title + beat);
        pages.push({
          kind: "beat",
          caption: beat,
          color: c,
          bw: b,
          indexLabel: String(i + 2).padStart(2, "0"),
        });
      });
      this.pages = pages;
      this.drawAt(0);
    }

    async loadUploadPages(files, title) {
      const pages = [];
      for (let i = 0; i < files.length; i++) {
        const url = URL.createObjectURL(files[i]);
        const img = await loadImage(url);
        const color = makeOffscreen();
        const ctx = color.getContext("2d");
        ctx.fillStyle = "#fffefb";
        ctx.fillRect(0, 0, W, H);
        const scale = Math.min(W / img.width, H / img.height);
        const dw = img.width * scale;
        const dh = img.height * scale;
        ctx.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
        const bw = makeOffscreen();
        desaturateTo(color, bw);
        pages.push({
          kind: i === 0 ? "cover" : "beat",
          caption: i === 0 ? title || "上传的故事" : "",
          color,
          bw,
          indexLabel: String(i + 1).padStart(2, "0"),
        });
      }
      this.pages = pages;
      this.drawAt(0);
    }

    pageAt(time) {
      const n = this.pages.length;
      if (!n) return { index: 0, local: 0, flip: 0 };
      const flip = this.transition === "page-flip" ? FLIP_SEC : 0;
      const span = this.pageSec + flip;
      let index = Math.floor(time / span);
      if (index >= n) return { index: n - 1, local: this.pageSec, flip: 0, done: true };
      const local = time - index * span;
      const flipP = flip && local > this.pageSec ? (local - this.pageSec) / flip : 0;
      return { index, local: Math.min(local, this.pageSec), flip: clamp(flipP, 0, 1) };
    }

    drawPageContent(targetCtx, page, local, mode) {
      targetCtx.fillStyle = "#fffefb";
      targetCtx.fillRect(0, 0, W, H);
      if (mode === "page-flip") {
        targetCtx.drawImage(page.color, 0, 0, W, H);
        drawCaption(targetCtx, page, 1);
        return;
      }
      targetCtx.drawImage(page.color, 0, 0, W, H);
      drawCaption(targetCtx, page, 1);
    }

    drawAt(time) {
      const ctx = this.ctx;
      ctx.fillStyle = "#fffefb";
      ctx.fillRect(0, 0, W, H);
      if (!this.pages.length) return;
      const pos = this.pageAt(time);
      const page = this.pages[pos.index];
      if (pos.flip > 0 && pos.index < this.pages.length - 1) {
        this.drawPageContent(this._fromBuf.getContext("2d"), page, this.pageSec, this.transition);
        this.drawPageContent(
          this._toBuf.getContext("2d"),
          this.pages[pos.index + 1],
          this.transition === "page-flip" ? this.pageSec : 0.01,
          this.transition
        );
        drawCurl(ctx, this._fromBuf, this._toBuf, pos.flip);
      } else {
        this.drawPageContent(ctx, page, pos.local, this.transition);
      }
      if (this.onFrame) this.onFrame(time, pos);
    }

    play() {
      if (this.playing) return;
      this.playing = true;
      this.t0 = performance.now() - this.elapsed * 1000;
      const tick = (now) => {
        if (!this.playing) return;
        this.elapsed = (now - this.t0) / 1000;
        if (this.elapsed >= this.totalSec) {
          this.elapsed = this.totalSec;
          this.drawAt(this.elapsed);
          this.playing = false;
          if (this.onEnded) this.onEnded();
          return;
        }
        this.drawAt(this.elapsed);
        this.raf = requestAnimationFrame(tick);
      };
      this.raf = requestAnimationFrame(tick);
    }

    pause() {
      this.playing = false;
      if (this.raf) cancelAnimationFrame(this.raf);
    }

    seek(t) {
      this.elapsed = clamp(t, 0, this.totalSec);
      if (this.playing) this.t0 = performance.now() - this.elapsed * 1000;
      this.drawAt(this.elapsed);
    }

    stop() {
      this.pause();
      this.elapsed = 0;
      this.drawAt(0);
    }

    playToEnd() {
      return new Promise((resolve) => {
        this.elapsed = 0;
        const prev = this.onEnded;
        this.onEnded = () => {
          this.onEnded = prev;
          if (prev) prev();
          resolve();
        };
        this.play();
      });
    }

    async recordBlob() {
      const stream = this.canvas.captureStream(30);
      const mime = MediaRecorder.isTypeSupported("video/webm;codecs=vp9")
        ? "video/webm;codecs=vp9"
        : MediaRecorder.isTypeSupported("video/webm;codecs=vp8")
          ? "video/webm;codecs=vp8"
          : "video/webm";
      const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 2800000 });
      const chunks = [];
      rec.ondataavailable = (e) => {
        if (e.data && e.data.size) chunks.push(e.data);
      };
      const stopped = new Promise((res) => {
        rec.onstop = res;
      });
      this.stop();
      rec.start(200);
      await new Promise((r) => setTimeout(r, 120));
      await this.playToEnd();
      await new Promise((r) => setTimeout(r, 200));
      rec.stop();
      await stopped;
      stream.getTracks().forEach((t) => t.stop());
      return new Blob(chunks, { type: mime });
    }
  }

  root.StoryPlayer = StoryPlayer;
  root.PLAYER_CONST = { W, H, PAGE_SEC, FLIP_SEC };
})(typeof globalThis !== "undefined" ? globalThis : this);
