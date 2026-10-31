(function (root, factory) {
  if (typeof module === "object" && module.exports) module.exports = factory();
  else root.StockJournalLedger = factory();
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  function tradeKey(t) {
    return `${t.market}:${String(t.symbol).toUpperCase()}`;
  }

  function pointValue(market, symbol) {
    if (market !== "NQ") return 1;
    const s = String(symbol || "").toUpperCase().replace(/[^A-Z]/g, "");
    if (s.indexOf("MNQ") === 0) return 2;
    return 20;
  }

  function round2(n) {
    return Math.round(Number(n) * 100) / 100;
  }

  function sortedTrades(list) {
    return [...list].sort((a, b) => {
      const da = `${a.date}T${a.time || "00:00"}`;
      const db = `${b.date}T${b.time || "00:00"}`;
      if (da !== db) return da < db ? -1 : 1;
      return String(a.createdAt || "").localeCompare(String(b.createdAt || ""));
    });
  }

  function calcFee(price, qty, market, settings) {
    if (!settings.autoFee) return 0;
    if (market === "TW") {
      const raw = Number(price) * Number(qty) * settings.feeRate * settings.feeDiscount;
      return Math.max(settings.minFee, Math.round(raw));
    }
    if (market === "CRYPTO") {
      const rate = settings.cryptoFeeRate != null ? settings.cryptoFeeRate : 0.001;
      return round2(Number(price) * Number(qty) * rate);
    }
    if (market === "NQ") {
      const per = settings.nqFeePerContract != null ? settings.nqFeePerContract : 0.62;
      return round2(Number(qty) * per);
    }
    return 0;
  }

  function calcTax(price, qty, side, market, settings) {
    if (!settings.autoFee || market !== "TW" || side !== "sell") return 0;
    return Math.round(Number(price) * Number(qty) * settings.taxRate);
  }

  function buildBooks(trades) {
    const books = new Map();
    const realizedByTrade = {};
    for (const t of sortedTrades(trades)) {
      const key = tradeKey(t);
      if (!books.has(key)) {
        books.set(key, {
          market: t.market,
          symbol: String(t.symbol).toUpperCase(),
          name: t.name || t.symbol,
          lots: [],
          realized: 0,
          buyQty: 0,
          sellQty: 0
        });
      }
      const book = books.get(key);
      if (t.name) book.name = t.name;
      const qty = Number(t.qty) || 0;
      const price = Number(t.price) || 0;
      const extra = (Number(t.fee) || 0) + (Number(t.tax) || 0);
      const pv = pointValue(t.market, t.symbol);
      if (qty <= 0) continue;
      let realizedHere = 0;
      if (t.side === "buy") {
        book.buyQty += qty;
        let left = qty;
        const costPer = price + extra / (qty * pv);
        while (left > 0 && book.lots.length && book.lots[0].qty < 0) {
          const lot = book.lots[0];
          const match = Math.min(left, -lot.qty);
          const pnl = (lot.costPerShare - costPer) * match * pv;
          book.realized += pnl;
          realizedHere += pnl;
          lot.qty += match;
          left -= match;
          if (Math.abs(lot.qty) < 1e-9) book.lots.shift();
        }
        if (left > 0) book.lots.push({ qty: left, costPerShare: costPer });
      } else {
        book.sellQty += qty;
        let left = qty;
        const netPer = price - extra / (qty * pv);
        while (left > 0 && book.lots.length && book.lots[0].qty > 0) {
          const lot = book.lots[0];
          const match = Math.min(left, lot.qty);
          const pnl = (netPer - lot.costPerShare) * match * pv;
          book.realized += pnl;
          realizedHere += pnl;
          lot.qty -= match;
          left -= match;
          if (Math.abs(lot.qty) < 1e-9) book.lots.shift();
        }
        if (left > 0) book.lots.push({ qty: -left, costPerShare: netPer });
      }
      realizedByTrade[t.id] = realizedHere;
    }
    return { books, realizedByTrade };
  }

  function positionsFrom(books, quotes) {
    const out = [];
    for (const book of books.values()) {
      let qty = 0;
      let weighted = 0;
      for (const lot of book.lots) {
        qty += lot.qty;
        weighted += lot.qty * lot.costPerShare;
      }
      const pv = pointValue(book.market, book.symbol);
      const avgCost = qty ? weighted / qty : 0;
      const cost = weighted * pv;
      const q = quotes[`${book.market}:${book.symbol}`];
      const quote = q ? Number(q.price) : null;
      const marketValue = quote != null ? quote * qty * pv : null;
      const unrealized = quote != null && qty ? (quote - avgCost) * qty * pv : null;
      out.push({
        market: book.market,
        symbol: book.symbol,
        name: book.name,
        qty,
        avgCost: Math.abs(avgCost),
        cost,
        quote,
        marketValue,
        unrealized,
        realized: book.realized,
        buyQty: book.buyQty,
        sellQty: book.sellQty,
        pointValue: pv
      });
    }
    return out.sort((a, b) => Math.abs(b.qty) - Math.abs(a.qty) || a.symbol.localeCompare(b.symbol, "zh-Hant"));
  }

  return { tradeKey, pointValue, sortedTrades, calcFee, calcTax, buildBooks, positionsFrom };
});
