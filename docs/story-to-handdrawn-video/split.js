(function (root) {
  "use strict";

  function splitStory(text) {
    const raw = String(text || "").replace(/\r\n/g, "\n").trim();
    if (!raw) return { title: "未命名故事", beats: [] };

    const lines = raw.split("\n").map((s) => s.trim()).filter(Boolean);
    let title = "未命名故事";
    let bodyLines = lines;

    if (lines.length >= 2) {
      const first = lines[0].replace(/^《+|》+$/g, "").trim();
      const looksLikeTitle =
        first.length <= 18 &&
        !/[。！？!?…]$/.test(first) &&
        !/[，,；;]/.test(first);
      if (looksLikeTitle) {
        title = first;
        bodyLines = lines.slice(1);
      }
    } else if (lines.length === 1 && lines[0].length <= 18 && !/[。！？!?]$/.test(lines[0])) {
      title = lines[0].replace(/^《+|》+$/g, "").trim();
      return { title, beats: [] };
    }

    const beats = [];
    for (const line of bodyLines) {
      const parts = line.match(/[^。！？!?]+[。！？!?]?/g) || [line];
      for (const part of parts) {
        const beat = part.trim();
        if (beat) beats.push(beat);
      }
    }
    return { title, beats };
  }

  function estimateDuration(pageCount, transition, pageSec) {
    const n = Math.max(0, pageCount | 0);
    if (n === 0) return 0;
    const hold = pageSec == null ? 4.8 : pageSec;
    const flip = transition === "page-flip" ? 0.7 : 0;
    return n * hold + Math.max(0, n - 1) * flip;
  }

  root.splitStory = splitStory;
  root.estimateDuration = estimateDuration;
  if (typeof module !== "undefined" && module.exports) {
    module.exports = { splitStory, estimateDuration };
  }
})(typeof globalThis !== "undefined" ? globalThis : this);
