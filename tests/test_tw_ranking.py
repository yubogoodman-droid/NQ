from __future__ import annotations

import json
import unittest

from tw.ranking import RankedStock, filter_by_price, filter_etfs, is_etf, parse_yahoo_ranking_html


def _html_with_table(rows: list[dict], rank_time: str = "2026-08-17T10:49:08+08:00") -> str:
    table = {
        "isFetching": False,
        "isFailed": False,
        "list": rows,
        "listMeta": {"rankTime": rank_time, "rankTimeRange": []},
        "pagination": {"resultsTotal": len(rows), "nextOffset": None},
        "params": {
            "exchange": "ALL",
            "limit": 100,
            "offset": 0,
            "sortBy": "-turnoverK",
            "period": "1D",
        },
    }
    payload = {"context": {"dispatcher": {"stores": {"TableStore": {"main-0-StockRanking": table}}}}}
    return (
        "<html><script>\n"
        "root.App || (root.App = {});\n"
        f"root.App.main = {json.dumps(payload, ensure_ascii=False)};\n"
        "</script></html>"
    )


class RankingTests(unittest.TestCase):
    def test_parse_and_price_filter(self) -> None:
        html = _html_with_table(
            [
                {
                    "rank": "1",
                    "name": "台積電",
                    "symbol": "2330.TW",
                    "price": "2400",
                    "change": "5",
                    "changePercent": "+0.21%",
                    "turnoverK": "10412850",
                    "volK": 4332,
                },
                {
                    "rank": "2",
                    "name": "南亞科",
                    "symbol": "2408.TW",
                    "price": "528",
                    "change": "16",
                    "changePercent": "+3.12%",
                    "turnoverK": "22556640",
                    "volK": 42948,
                },
                {
                    "rank": "3",
                    "name": "合晶",
                    "symbol": "6182.TWO",
                    "price": "122.5",
                    "change": "11",
                    "changePercent": "+9.87%",
                    "turnoverK": "9281310",
                    "volK": 79073,
                },
            ]
        )
        stocks, rank_time = parse_yahoo_ranking_html(html)
        self.assertEqual(rank_time, "2026-08-17T10:49:08+08:00")
        self.assertEqual([s.symbol for s in stocks], ["2330.TW", "2408.TW", "6182.TWO"])
        self.assertEqual(stocks[0].turnover, 10412850 * 1000)
        self.assertEqual(stocks[2].exchange, "TWO")

        kept = filter_by_price(stocks, 650)
        self.assertEqual([s.symbol for s in kept], ["2408.TW", "6182.TWO"])
        self.assertTrue(all(s.price < 650 for s in kept))

    def test_filter_excludes_price_equal_to_limit(self) -> None:
        stocks = [
            RankedStock(1, "9999.TW", "測", 650.0, None, None, 1, 1.0, "TAI"),
            RankedStock(2, "1111.TW", "測2", 649.9, None, None, 1, 1.0, "TAI"),
        ]
        kept = filter_by_price(stocks, 650)
        self.assertEqual([s.symbol for s in kept], ["1111.TW"])

    def test_filter_excludes_etfs(self) -> None:
        stocks = [
            RankedStock(1, "0050.TW", "元大台灣50", 106.0, None, None, 1, 1.0, "TAI"),
            RankedStock(2, "00631L.TW", "元大台灣50正2", 36.0, None, None, 1, 1.0, "TAI"),
            RankedStock(3, "00981A.TW", "主動統一台股增長", 30.0, None, None, 1, 1.0, "TAI"),
            RankedStock(4, "00878.TW", "國泰永續高股息", 33.0, None, None, 1, 1.0, "TAI"),
            RankedStock(5, "2327.TW", "國巨", 611.0, None, None, 1, 1.0, "TAI"),
            RankedStock(6, "4958.TW", "臻鼎-KY", 504.0, None, None, 1, 1.0, "TAI"),
        ]
        self.assertTrue(is_etf(stocks[0]))
        self.assertTrue(is_etf(stocks[1]))
        self.assertTrue(is_etf(stocks[2]))
        self.assertTrue(is_etf(stocks[3]))
        self.assertFalse(is_etf(stocks[4]))
        self.assertFalse(is_etf(stocks[5]))
        kept = filter_etfs(stocks)
        self.assertEqual([s.symbol for s in kept], ["2327.TW", "4958.TW"])


if __name__ == "__main__":
    unittest.main()
