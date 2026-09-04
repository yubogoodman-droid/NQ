const { buildBooks, positionsFrom, calcFee, calcTax, pointValue } = require("./ledger.js");

const settings = {
  feeRate: 0.001425,
  feeDiscount: 0.6,
  minFee: 20,
  taxRate: 0.003,
  autoFee: true,
  cryptoFeeRate: 0.001,
  nqFeePerContract: 0.62
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

const media = buildBooks([
  { id: "m1", date: "2026-06-15", time: "10:02", market: "TW", symbol: "2454", name: "聯發科", side: "buy", price: 1280, qty: 500, fee: calcFee(1280, 500, "TW", settings), tax: 0 },
  { id: "m2", date: "2026-08-01", time: "13:33", market: "TW", symbol: "2454", name: "聯發科", side: "sell", price: 1190, qty: 500, fee: calcFee(1190, 500, "TW", settings), tax: calcTax(1190, 500, "sell", "TW", settings) }
]);
const mediaPos = positionsFrom(media.books, {})[0];
almost(mediaPos.qty, 0, 1e-6, "2454 closed");
assert(media.realizedByTrade.m2 < 0, "2454 sell is a loss");
almost(mediaPos.realized, -47841, 1, "2454 closed realized");

const shorts = buildBooks([
  { id: "s1", date: "2026-01-01", time: "09:00", market: "TW", symbol: "2603", side: "sell", price: 200, qty: 1000, fee: 0, tax: 600 },
  { id: "s2", date: "2026-01-02", time: "09:00", market: "TW", symbol: "2603", side: "buy", price: 180, qty: 1000, fee: 0, tax: 0 }
]);
const shortPos = positionsFrom(shorts.books, {})[0];
almost(shortPos.qty, 0, 1e-6, "covered short");
almost(shorts.realizedByTrade.s2, (200 - 0.6 - 180) * 1000, 0.02, "short cover pnl");

assert(pointValue("NQ", "NQ") === 20, "NQ point");
assert(pointValue("NQ", "MNQ") === 2, "MNQ point");
assert(pointValue("NQ", "MNQ1!") === 2, "MNQ1 point");
assert(pointValue("CRYPTO", "BTC") === 1, "crypto point");

const btcFeeIn = calcFee(65000, 0.1, "CRYPTO", settings);
const btcFeeOut = calcFee(70000, 0.1, "CRYPTO", settings);
almost(btcFeeIn, 6.5, 0.001, "btc fee in");
almost(btcFeeOut, 7, 0.001, "btc fee out");
const btc = buildBooks([
  { id: "b1", date: "2026-02-01", time: "08:00", market: "CRYPTO", symbol: "BTC", name: "比特幣", side: "buy", price: 65000, qty: 0.1, fee: btcFeeIn, tax: 0 },
  { id: "b2", date: "2026-02-10", time: "08:00", market: "CRYPTO", symbol: "BTC", name: "比特幣", side: "sell", price: 70000, qty: 0.1, fee: btcFeeOut, tax: 0 }
]);
almost(btc.realizedByTrade.b2, (70000 - btcFeeOut / 0.1 - 65000 - btcFeeIn / 0.1) * 0.1, 0.02, "btc realized");

const mnqFee = calcFee(19200, 2, "NQ", settings);
almost(mnqFee, 1.24, 0.001, "mnq fee");
const mnq = buildBooks([
  { id: "n1", date: "2026-03-01", time: "21:00", market: "NQ", symbol: "MNQ", name: "納斯達克微台", side: "buy", price: 19200, qty: 2, fee: mnqFee, tax: 0 },
  { id: "n2", date: "2026-03-02", time: "22:00", market: "NQ", symbol: "MNQ", name: "納斯達克微台", side: "sell", price: 19450, qty: 2, fee: mnqFee, tax: 0 }
]);
almost(mnq.realizedByTrade.n2, 250 * 2 * 2 - mnqFee - mnqFee, 0.05, "mnq realized uses $2 point");
const mnqOpen = buildBooks([
  { id: "n3", date: "2026-03-01", time: "21:00", market: "NQ", symbol: "NQ", name: "納斯達克小台", side: "buy", price: 20000, qty: 1, fee: 0, tax: 0 }
]);
const nqPos = positionsFrom(mnqOpen.books, { "NQ:NQ": { price: 20100 } })[0];
almost(nqPos.unrealized, 100 * 20, 0.02, "NQ unrealized uses $20 point");
almost(nqPos.marketValue, 20100 * 20, 0.02, "NQ notional");

console.log("ledger tests passed");
