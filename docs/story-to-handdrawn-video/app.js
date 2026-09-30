(function () {
  "use strict";

  const DEMO_TEXT = `窗边的约定

小猫坐在窗边，看着下雨的街道。
一只小鸟停在湿漉漉的枝头，抖了抖羽毛。
小猫轻轻抬起爪子，隔着玻璃打招呼。
小鸟歪了歪头，好像听懂了。
雨停以后，阳光把窗台晒得暖暖的。`;

  const PRESETS = [
    { id: "window", title: "窗边的约定", art: "ai", text: DEMO_TEXT },
    {
      id: "moon",
      title: "月亮邮差",
      art: "doodle",
      text: `月亮邮差

夜空里，小邮差骑着云朵出发了。
他把一封信轻轻放在窗台上。
信里写着：明天会天晴。
孩子做了一个很甜的梦。`,
    },
    {
      id: "soup",
      title: "一碗热汤",
      art: "doodle",
      text: `一碗热汤

外婆把热汤端到桌上。
蒸汽慢慢升起来。
小孩吹了又吹，终于喝了一口。
窗外的雨，好像也变小了。`,
    },
  ];

  const $ = (id) => document.getElementById(id);
  const canvas = $("stage");
  const player = new StoryPlayer(canvas);
  let demoStory = null;
  let source = "ai";
  let uploaded = [];

  function normalize(s) {
    return String(s || "")
      .replace(/\s+/g, "")
      .trim();
  }

  function fmt(sec) {
    const s = Math.max(0, sec);
    const m = Math.floor(s / 60);
    const r = Math.floor(s % 60);
    return m + ":" + String(r).padStart(2, "0");
  }

  function setStatus(msg) {
    $("status").textContent = msg;
  }

  function updateMeta() {
    const n = player.pages.length;
    $("pageCount").textContent = n ? n + " 页" : "—";
    $("durationLabel").textContent = n ? fmt(player.totalSec) : "0:00";
    $("timeLabel").textContent = fmt(player.elapsed);
    $("seek").max = String(player.totalSec || 0);
    $("seek").value = String(player.elapsed);
    $("playBtn").textContent = player.playing ? "暂停" : "播放";
    renderDots(player.pageAt(player.elapsed).index);
  }

  function renderDots(active) {
    const el = $("dots");
    el.innerHTML = player.pages
      .map(
        (p, i) =>
          `<button type="button" class="dot${i === active ? " on" : ""}" data-i="${i}" aria-label="第 ${i + 1} 页"></button>`
      )
      .join("");
  }

  player.onFrame = () => updateMeta();
  player.onEnded = () => {
    $("playBtn").textContent = "播放";
    setStatus("播放结束，可下载成片或换一个故事。");
    updateMeta();
  };

  function currentStyle() {
    return $("styleSel").value;
  }
  function currentTransition() {
    return $("transSel").value;
  }

  async function loadDemo() {
    if (!demoStory) {
      const res = await fetch("story.json");
      demoStory = await res.json();
    }
    $("storyInput").value = demoStory.text;
    player.transition = currentTransition();
    source = "ai";
    await player.loadAiPages(demoStory);
    updateMeta();
    setStatus("示例故事已载入：彩铅日记漫画 · 文字 → 黑白 → 彩色。");
  }

  function generateFromText() {
    const text = $("storyInput").value;
    const { title, beats } = splitStory(text);
    if (!beats.length) {
      setStatus("请先贴上故事，每句一行或用句号分开。");
      return;
    }
    player.transition = currentTransition();
    if (normalize(text) === normalize(DEMO_TEXT) && demoStory) {
      source = "ai";
      return player.loadAiPages(demoStory).then(() => {
        updateMeta();
        setStatus("使用锁定的彩铅日记插画。");
      });
    }
    source = "doodle";
    player.loadDoodlePages(title, beats, currentStyle());
    updateMeta();
    setStatus("已按句子分镜，画布手绘 " + (beats.length + 1) + " 页。");
  }

  async function generateFromUpload(files) {
    if (!files || !files.length) return;
    uploaded = Array.from(files);
    const { title } = splitStory($("storyInput").value);
    player.transition = currentTransition();
    source = "upload";
    await player.loadUploadPages(uploaded, title);
    updateMeta();
    setStatus("已按上传顺序导入 " + uploaded.length + " 张，构图 contain 不裁切。");
  }

  $("playBtn").addEventListener("click", () => {
    if (!player.pages.length) return;
    if (player.playing) player.pause();
    else player.play();
    updateMeta();
  });
  $("stopBtn").addEventListener("click", () => {
    player.stop();
    updateMeta();
  });
  $("seek").addEventListener("input", (e) => {
    player.seek(Number(e.target.value));
    updateMeta();
  });
  $("dots").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-i]");
    if (!btn) return;
    const i = Number(btn.dataset.i);
    const flip = currentTransition() === "page-flip" ? 0.7 : 0;
    player.seek(i * (player.pageSec + flip));
    updateMeta();
  });

  $("styleSel").addEventListener("change", () => {
    if (source === "doodle") generateFromText();
  });
  $("transSel").addEventListener("change", () => {
    player.transition = currentTransition();
    player.drawAt(player.elapsed);
    updateMeta();
  });

  $("genBtn").addEventListener("click", () => {
    generateFromText();
  });
  $("demoBtn").addEventListener("click", () => {
    loadDemo().then(() => {
      player.stop();
      player.play();
    });
  });

  $("fileInput").addEventListener("change", (e) => generateFromUpload(e.target.files));

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = PRESETS.find((x) => x.id === btn.dataset.preset);
      if (!p) return;
      $("storyInput").value = p.text;
      document.querySelectorAll("[data-preset]").forEach((b) => b.classList.toggle("on", b === btn));
      if (p.art === "ai") loadDemo();
      else generateFromText();
    });
  });

  $("dlBtn").addEventListener("click", async () => {
    if (!player.pages.length) return;
    $("dlBtn").disabled = true;
    setStatus("正在录制成片（静音 WebM）…请等这一遍播完。");
    try {
      const blob = await player.recordBlob();
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      const { title } = splitStory($("storyInput").value);
      a.download = (title || "handdrawn-story") + ".webm";
      a.click();
      setStatus("已下载静音画面轨，可后期配音。体积约 " + Math.round(blob.size / 1024) + " KB。");
    } catch (err) {
      setStatus("录制失败：" + (err && err.message ? err.message : err));
    } finally {
      $("dlBtn").disabled = false;
      updateMeta();
    }
  });

  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("playBtn").click();
    }
  });

  Promise.all([document.fonts ? document.fonts.ready : Promise.resolve(), loadDemo()]).catch((err) => {
    setStatus("载入示例失败：" + err.message);
  });
})();
