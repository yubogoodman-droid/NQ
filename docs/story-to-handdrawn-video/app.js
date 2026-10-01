(function () {
  "use strict";

  const LIUTI_TEXT = `留體力學

小明是個習武之人。
有一天一位大師要傳授畢生絕學給小明。
小明便開始苦練基礎。
每天到筋疲力盡為止。
在傳授那天，大師對著小明說：「你這呆徒」。
不留一點體力，我要怎麼把功夫傳給你？
小明：「可是師父，留體力學，好難」。`;

  const TREE_TEXT = `大樹跟小樹

大樹跟小樹。
大樹跟小樹差在那裏？
答案是插在土裡。`;

  const WINDOW_TEXT = `窗边的约定

小猫坐在窗边，看着下雨的街道。
一只小鸟停在湿漉漉的枝头，抖了抖羽毛。
小猫轻轻抬起爪子，隔着玻璃打招呼。
小鸟歪了歪头，好像听懂了。
雨停以后，阳光把窗台晒得暖暖的。`;

  const REMOTION = {
    liuti: {
      video: "assets/liuti-preview.mp4",
      poster: "assets/liuti-poster.jpg",
      download: "留體力學.mp4",
      sheet: "assets/liuti-character-sheet.jpg",
      sheetAlt: "小明與大師角色設定",
      pages: 7,
      duration: 15.0,
      storyUrl: "liuti.json",
      colorOnly: true,
    },
    trees: {
      video: "assets/tree-riddle-preview.mp4",
      poster: "assets/tree-poster.jpg",
      download: "大樹跟小樹.mp4",
      sheet: "assets/tree-character-sheet.jpg",
      sheetAlt: "大樹與小樹角色設定",
      pages: 3,
      duration: 13.7,
      storyUrl: "trees.json",
      colorOnly: true,
    },
    window: {
      video: "assets/demo-preview.mp4",
      poster: "assets/scenes/00-cover.jpg",
      download: "窗边的约定.mp4",
      sheet: "assets/character-sheet.jpg",
      sheetAlt: "小猫与小鸟角色设定",
      pages: 6,
      duration: 30.6,
      storyUrl: "story.json",
    },
  };

  const PRESETS = [
    { id: "liuti", title: "留體力學", art: "ai", remotion: "liuti", text: LIUTI_TEXT },
    { id: "trees", title: "大樹跟小樹", art: "ai", remotion: "trees", text: TREE_TEXT },
    { id: "window", title: "窗边的约定", art: "ai", remotion: "window", text: WINDOW_TEXT },
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
  const video = $("demoVideo");
  const book = $("book");
  const player = new StoryPlayer(canvas);
  let demoStories = {};
  let activeRemotion = "liuti";
  let source = "ai";
  let uploaded = [];
  let view = "video";

  function setView(mode) {
    view = mode;
    book.classList.toggle("mode-video", mode === "video");
    book.classList.toggle("mode-canvas", mode !== "video");
    if (mode !== "video") {
      video.pause();
    }
  }

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

  function remotionMeta() {
    return REMOTION[activeRemotion] || REMOTION.liuti;
  }

  function updateMeta() {
    if (view === "video") {
      const demo = remotionMeta();
      const dur = video.duration && isFinite(video.duration) ? video.duration : demo.duration;
      $("pageCount").textContent = demo.pages + " 页 · Remotion";
      $("durationLabel").textContent = fmt(dur);
      $("timeLabel").textContent = fmt(video.currentTime || 0);
      $("seek").max = String(dur);
      $("seek").value = String(video.currentTime || 0);
      $("playBtn").textContent = video.paused ? "播放" : "暂停";
      const pageSec = dur / Math.max(1, demo.pages);
      const page = Math.min(demo.pages - 1, Math.floor((video.currentTime || 0) / pageSec));
      renderDots(page, demo.pages);
      return;
    }
    const n = player.pages.length;
    $("pageCount").textContent = n ? n + " 页" : "—";
    $("durationLabel").textContent = n ? fmt(player.totalSec) : "0:00";
    $("timeLabel").textContent = fmt(player.elapsed);
    $("seek").max = String(player.totalSec || 0);
    $("seek").value = String(player.elapsed);
    $("playBtn").textContent = player.playing ? "暂停" : "播放";
    renderDots(player.pageAt(player.elapsed).index, n);
  }

  function renderDots(active, count) {
    const n = count == null ? player.pages.length : count;
    const el = $("dots");
    el.innerHTML = Array.from({ length: n }, (_, i) =>
      `<button type="button" class="dot${i === active ? " on" : ""}" data-i="${i}" aria-label="第 ${i + 1} 页"></button>`
    ).join("");
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

  function applyRemotionChrome(id) {
    const demo = REMOTION[id];
    if (!demo) return;
    activeRemotion = id;
    if (video.getAttribute("src") !== demo.video) {
      video.pause();
      video.src = demo.video;
      video.poster = demo.poster;
      video.load();
    }
    const dl = $("demoDl");
    if (dl) {
      dl.href = demo.video;
      dl.setAttribute("download", demo.download);
    }
    const img = $("castImg");
    if (img) {
      img.src = demo.sheet;
      img.alt = demo.sheetAlt;
    }
  }

  async function loadRemotion(id) {
    const demo = REMOTION[id] || REMOTION.liuti;
    applyRemotionChrome(id);
    if (!demoStories[id] && demo.storyUrl) {
      const res = await fetch(demo.storyUrl);
      demoStories[id] = await res.json();
    }
    const story = demoStories[id];
    $("storyInput").value = (story && story.text) || PRESETS.find((p) => p.remotion === id).text;
    player.transition = currentTransition();
    source = "ai";
    setView("video");
    if (story && story.pages && story.pages[0] && story.pages[0].color) {
      await player.loadAiPages(story);
    }
    updateMeta();
    const note = demo.colorOnly
      ? "彩图直出，无黑白上色。"
      : "文字 → 黑白 → 彩色。";
    setStatus("GitHub 源项目 Remotion 成片已载入：" + note);
  }

  function generateFromText() {
    const text = $("storyInput").value;
    const { title, beats } = splitStory(text);
    if (!beats.length) {
      setStatus("请先贴上故事，每句一行或用句号分开。");
      return;
    }
    player.transition = currentTransition();
    const remotionPreset = PRESETS.find(
      (p) => p.art === "ai" && normalize(text) === normalize(p.text)
    );
    if (remotionPreset) {
      source = "ai";
      setView("video");
      return loadRemotion(remotionPreset.remotion).then(() => {
        setStatus("使用 GitHub Remotion 示例成片。");
      });
    }
    source = "doodle";
    player.pause();
    setView("canvas");
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
    player.pause();
    setView("canvas");
    await player.loadUploadPages(uploaded, title);
    updateMeta();
    setStatus("已按上传顺序导入 " + uploaded.length + " 张，构图 contain 不裁切。");
  }

  $("playBtn").addEventListener("click", () => {
    if (view === "video") {
      if (video.paused) video.play();
      else video.pause();
      updateMeta();
      return;
    }
    if (!player.pages.length) return;
    if (player.playing) player.pause();
    else player.play();
    updateMeta();
  });
  $("stopBtn").addEventListener("click", () => {
    if (view === "video") {
      video.pause();
      video.currentTime = 0;
      updateMeta();
      return;
    }
    player.stop();
    updateMeta();
  });
  $("seek").addEventListener("input", (e) => {
    const t = Number(e.target.value);
    if (view === "video") {
      video.currentTime = t;
      updateMeta();
      return;
    }
    player.seek(t);
    updateMeta();
  });
  $("dots").addEventListener("click", (e) => {
    const btn = e.target.closest("[data-i]");
    if (!btn) return;
    const i = Number(btn.dataset.i);
    if (view === "video") {
      const demo = remotionMeta();
      const dur = video.duration && isFinite(video.duration) ? video.duration : demo.duration;
      video.currentTime = i * (dur / Math.max(1, demo.pages));
      updateMeta();
      return;
    }
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
    loadRemotion(activeRemotion).then(() => {
      video.currentTime = 0;
      video.play();
      updateMeta();
    });
  });

  $("fileInput").addEventListener("change", (e) => generateFromUpload(e.target.files));

  document.querySelectorAll("[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const p = PRESETS.find((x) => x.id === btn.dataset.preset);
      if (!p) return;
      $("storyInput").value = p.text;
      document.querySelectorAll("[data-preset]").forEach((b) => b.classList.toggle("on", b === btn));
      if (p.art === "ai") loadRemotion(p.remotion);
      else generateFromText();
    });
  });

  $("dlBtn").addEventListener("click", async () => {
    if (view === "video") {
      const a = $("demoDl");
      if (a) a.click();
      return;
    }
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

  video.addEventListener("timeupdate", () => {
    if (view === "video") updateMeta();
  });
  video.addEventListener("loadedmetadata", () => {
    if (view === "video") updateMeta();
  });

  document.addEventListener("keydown", (e) => {
    if (e.target && (e.target.tagName === "TEXTAREA" || e.target.tagName === "INPUT")) return;
    if (e.code === "Space") {
      e.preventDefault();
      $("playBtn").click();
    }
  });

  Promise.all([document.fonts ? document.fonts.ready : Promise.resolve(), loadRemotion("liuti")]).catch((err) => {
    setStatus("载入示例失败：" + err.message);
  });
})();
