const { buildBooks, positionsFrom, calcFee, calcTax } = require("./ledger.js");

const settings = {
  feeRate: 0.001425,
  feeDiscount: 0.6,
  minFee: 20,
  taxRate: 0.003,
  autoFee: true
};

function assert(cond, msg) {
  if (!cond) throw new Error(msg);
}
function almost(a, b, eps, msg) {
  if (Math.abs(a - b) > (eps == null ? 0.02 : eps)) {
    throw new Error(`${msg}: ${a} !== ${b}`);
  }
}

const fee890 = calcFee(890, 1000, "TW", settings);
const fee980 = calcFee(980, 1000, "TW", settings);
const fee1050 = calcFee(1050, 1000, "TW", settings);
const tax1050 = calcTax(1050, 1000, "sell", "TW", settings);
almost(fee890, 761, 0, "fee 890");
almost(fee980, 838, 0, "fee 980");
almost(fee1050, 898, 0, "fee 1050");
almost(tax1050, 3150, 0, "tax 1050");
assert(calcTax(1050, 1000, "buy", "TW", settings) === 0, "buy has no tax");
assert(calcFee(100, 1, "US", settings) === 0, "US fee off");

const trades = [
  { id: "1", date: "2026-03-12", time: "10:21", market: "TW", symbol: "2330", name: "台積電", side: "buy", price: 890, qty: 1000, fee: fee890, tax: 0 },
  { id: "2", date: "2026-05-20", time: "13:05", market: "TW", symbol: "2330", name: "台積電", side: "buy", price: 980, qty: 1000, fee: fee980, tax: 0 },
  { id: "3", date: "2026-07-08", time: "09:48", market: "TW", symbol: "2330", name: "台積電", side: "sell", price: 1050, qty: 1000, fee: fee1050, tax: tax1050 }
];

const { books, realizedByTrade } = buildBooks(trades);
const pos = positionsFrom(books, { "TW:2330": { price: 960 } })[0];

almost(pos.qty, 1000, 1e-6, "remaining qty");
almost(pos.avgCost, 980 + fee980 / 1000, 1e-6, "fifo leftover is second lot");
almost(realizedByTrade["3"], (1050 - (fee1050 + tax1050) / 1000 - (890 + fee890 / 1000)) * 1000, 0.02, "realized sell");
almost(pos.unrealized, (960 - pos.avgCost) * 1000, 0.02, "unrealized");
assert(realizedByTrade["1"] === 0 && realizedByTrade["2"] === 0, "buys realize 0");

const shorts = buildBooks([
  { id: "s1", date: "2026-01-01", time: "09:00", market: "TW", symbol: "2603", side: "sell", price: 200, qty: 1000, fee: 0, tax: 600 },
  { id: "s2", date: "2026-01-02", time: "09:00", market: "TW", symbol: "2603", side: "buy", price: 180, qty: 1000, fee: 0, tax: 0 }
]);
const shortPos = positionsFrom(shorts.books, {})[0];
almost(shortPos.qty, 0, 1e-6, "covered short");
almost(shorts.realizedByTrade.s2, (200 - 0.6 - 180) * 1000, 0.02, "short cover pnl");

console.log("ledger tests passed");
