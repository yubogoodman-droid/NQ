from __future__ import annotations

import json
import unittest

from datetime import date
from unittest.mock import patch

from tw.ranking import (
    RankedStock,
    fetch_daily_turnover_ranking,
    fetch_turnover_ranking,
    filter_by_price,
    filter_etfs,
    filter_financials,
    filter_telecoms,
    is_etf,
    is_financial,
    is_telecom,
    iter_recent_sessions,
    parse_tpex_quotes,
    parse_twse_mi_index,
    parse_yahoo_ranking_html,
    last_n_weekdays,
    previous_friday,
    previous_weekdays,
)


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

    def test_filter_excludes_financials(self) -> None:
        stocks = [
            RankedStock(1, "2884.TW", "玉山金", 30.0, None, None, 1, 1.0, "TAI"),
            RankedStock(2, "2887.TW", "台新新光金", 20.0, None, None, 1, 1.0, "TAI"),
            RankedStock(3, "2801.TW", "彰化銀行", 20.0, None, None, 1, 1.0, "TAI"),
            RankedStock(4, "5871.TW", "中租-KY", 150.0, None, None, 1, 1.0, "TAI"),
            RankedStock(5, "6005.TWO", "群益證", 18.0, None, None, 1, 1.0, "TWO"),
            RankedStock(6, "3653.TW", "金居", 420.0, None, None, 1, 1.0, "TAI"),
            RankedStock(7, "2312.TW", "金寶", 40.0, None, None, 1, 1.0, "TAI"),
            RankedStock(8, "2327.TW", "國巨*", 611.0, None, None, 1, 1.0, "TAI"),
            RankedStock(9, "4958.TW", "臻鼎-KY", 504.0, None, None, 1, 1.0, "TAI"),
        ]
        self.assertTrue(is_financial(stocks[0]))
        self.assertTrue(is_financial(stocks[1]))
        self.assertTrue(is_financial(stocks[2]))
        self.assertTrue(is_financial(stocks[3]))
        self.assertTrue(is_financial(stocks[4]))
        self.assertFalse(is_financial(stocks[5]))
        self.assertFalse(is_financial(stocks[6]))
        self.assertFalse(is_financial(stocks[7]))
        self.assertFalse(is_financial(stocks[8]))
        kept = filter_financials(stocks)
        self.assertEqual(
            [s.symbol for s in kept],
            ["3653.TW", "2312.TW", "2327.TW", "4958.TW"],
        )

    def test_filter_excludes_telecoms(self) -> None:
        stocks = [
            RankedStock(1, "2412.TW", "中華電", 130.0, None, None, 1, 1.0, "TAI"),
            RankedStock(2, "3045.TW", "台灣大", 110.0, None, None, 1, 1.0, "TAI"),
            RankedStock(3, "4904.TW", "遠傳", 90.0, None, None, 1, 1.0, "TAI"),
            RankedStock(4, "3682.TW", "亞太電", 10.0, None, None, 1, 1.0, "TAI"),
            RankedStock(5, "2345.TW", "智邦", 700.0, None, None, 1, 1.0, "TAI"),
            RankedStock(6, "2303.TW", "聯電", 50.0, None, None, 1, 1.0, "TAI"),
            RankedStock(7, "2327.TW", "國巨*", 611.0, None, None, 1, 1.0, "TAI"),
        ]
        self.assertTrue(is_telecom(stocks[0]))
        self.assertTrue(is_telecom(stocks[1]))
        self.assertTrue(is_telecom(stocks[2]))
        self.assertTrue(is_telecom(stocks[3]))
        self.assertFalse(is_telecom(stocks[4]))
        self.assertFalse(is_telecom(stocks[5]))
        self.assertFalse(is_telecom(stocks[6]))
        kept = filter_telecoms(stocks)
        self.assertEqual(
            [s.symbol for s in kept],
            ["2345.TW", "2303.TW", "2327.TW"],
        )

    def test_previous_friday_from_monday(self) -> None:
        self.assertEqual(previous_friday(date(2026, 8, 17)), date(2026, 8, 14))
        self.assertEqual(previous_friday(date(2026, 8, 14)), date(2026, 8, 7))

    def test_last_n_weekdays_includes_today_if_weekday(self) -> None:
        self.assertEqual(
            last_n_weekdays(5, date(2026, 8, 21)),
            [
                date(2026, 8, 17),
                date(2026, 8, 18),
                date(2026, 8, 19),
                date(2026, 8, 20),
                date(2026, 8, 21),
            ],
        )
        self.assertEqual(
            last_n_weekdays(2, date(2026, 8, 16)),
            [date(2026, 8, 13), date(2026, 8, 14)],
        )

    def test_previous_weekdays_is_last_mon_to_fri(self) -> None:
        self.assertEqual(
            previous_weekdays(date(2026, 8, 17)),
            [
                date(2026, 8, 10),
                date(2026, 8, 11),
                date(2026, 8, 12),
                date(2026, 8, 13),
                date(2026, 8, 14),
            ],
        )
        self.assertEqual(previous_weekdays(date(2026, 8, 12))[0], date(2026, 8, 3))

    def test_previous_weekdays_two_weeks(self) -> None:
        self.assertEqual(
            previous_weekdays(date(2026, 8, 17), weeks=2),
            [
                date(2026, 8, 3),
                date(2026, 8, 4),
                date(2026, 8, 5),
                date(2026, 8, 6),
                date(2026, 8, 7),
                date(2026, 8, 10),
                date(2026, 8, 11),
                date(2026, 8, 12),
                date(2026, 8, 13),
                date(2026, 8, 14),
            ],
        )

    def test_parse_twse_and_tpex_daily(self) -> None:
        twse = {
            "tables": [
                {
                    "fields": ["證券代號", "證券名稱", "成交股數", "成交筆數", "成交金額", "開盤價", "最高價", "最低價", "收盤價", "漲跌(+/-)", "漲跌價差"],
                    "data": [
                        ["2330", "台積電", "10,000,000", "1", "12,000,000,000", "1200", "1210", "1190", "1200", "<p style='color:green'>-</p>", "10"],
                        ["2408", "南亞科", "1,000,000", "1", "500,000,000", "500", "510", "490", "500", "<p style='color:red'>+</p>", "5"],
                    ],
                }
            ]
        }
        tpex = {
            "tables": [
                {
                    "fields": ["代號", "名稱", "收盤 ", "漲跌", "開盤 ", "最高 ", "最低", "成交股數  ", " 成交金額(元)"],
                    "data": [
                        ["6182", "合晶", "122.5", "+11", "110", "123", "110", "7,900,000", "900,000,000"],
                    ],
                }
            ]
        }
        listed = parse_twse_mi_index(twse)
        otc = parse_tpex_quotes(tpex)
        self.assertEqual(listed[0].symbol, "2330.TW")
        self.assertEqual(listed[0].change, -10.0)
        self.assertEqual(listed[1].symbol, "2408.TW")
        self.assertEqual(listed[1].change, 5.0)
        self.assertEqual(otc[0].symbol, "6182.TWO")
        self.assertEqual(otc[0].turnover, 900_000_000)
        self.assertAlmostEqual(otc[0].change_percent or 0, 11 / 111.5 * 100, places=4)

    def test_iter_recent_sessions_skips_weekend(self) -> None:
        days = list(iter_recent_sessions(date(2026, 8, 16), limit=3))
        self.assertEqual(days, [date(2026, 8, 14), date(2026, 8, 13), date(2026, 8, 12)])

    def test_live_ranking_falls_back_to_previous_session(self) -> None:
        today = date(2026, 8, 20)
        rows = [
            RankedStock(i, f"{1000 + i}.TW", "測", 10.0, None, None, 1, 1e9 - i, "TAI")
            for i in range(60)
        ]

        def fake_daily(on_date, top=100, session=None, timeout=20):
            if on_date == today:
                raise ValueError("not published")
            return rows[:top], f"{on_date.isoformat()} 盤後成交額"

        with patch("tw.ranking.fetch_daily_turnover_ranking", side_effect=fake_daily):
            ranked, label = fetch_turnover_ranking(as_of=today)
        self.assertEqual(len(ranked), 60)
        self.assertIn("2026-08-19", label)
        self.assertIn("尚未公布", label)

    def test_daily_ranking_keeps_one_market_if_other_fails(self) -> None:
        rows = [
            RankedStock(i, f"{2000 + i}.TW", "測", 20.0, None, None, 1, 8e8 - i, "TAI")
            for i in range(40)
        ]
        with (
            patch("tw.ranking._fetch_twse_daily", return_value=rows),
            patch("tw.ranking._fetch_tpex_daily", side_effect=ValueError("down")),
        ):
            ranked, label = fetch_daily_turnover_ranking(date(2026, 8, 19), top=100)
        self.assertEqual(len(ranked), 40)
        self.assertIn("2026-08-19", label)


if __name__ == "__main__":
    unittest.main()
