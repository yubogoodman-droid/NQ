const { splitStory, estimateDuration } = require("./split.js");

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}

const empty = splitStory("   ");
assert(empty.title === "未命名故事", "empty title");
assert(empty.beats.length === 0, "empty beats");

const demo = splitStory(`窗边的约定

小猫坐在窗边，看着下雨的街道。
一只小鸟停在湿漉漉的枝头，抖了抖羽毛。
小猫轻轻抬起爪子，隔着玻璃打招呼。
小鸟歪了歪头，好像听懂了。
雨停以后，阳光把窗台晒得暖暖的。`);

assert(demo.title === "窗边的约定", "demo title");
assert(demo.beats.length === 5, "demo has 5 beats, got " + demo.beats.length);
assert(demo.beats[0] === "小猫坐在窗边，看着下雨的街道。", "first sentence kept");
assert(demo.beats[4].includes("暖暖的"), "last sentence kept");

const noPunct = splitStory("夜航\n云朵出发了\n信放在窗台");
assert(noPunct.title === "夜航", "newline title");
assert(noPunct.beats.length === 2, "newline beats");

const oneLine = splitStory("小猫坐在窗边。小鸟停在枝头。");
assert(oneLine.title === "未命名故事", "no title when only sentences");
assert(oneLine.beats.length === 2, "two sentences one line");

const bookTitle = splitStory("《月亮邮差》\n夜空里，小邮差出发了。");
assert(bookTitle.title === "月亮邮差", "strip book marks");

assert(estimateDuration(0, "cut") === 0, "zero pages");
assert(Math.abs(estimateDuration(6, "cut", 4.8) - 28.8) < 1e-6, "cut duration");
assert(Math.abs(estimateDuration(6, "page-flip", 4.8) - (28.8 + 3.5)) < 1e-6, "flip duration");

console.log("split.test.js ok");
