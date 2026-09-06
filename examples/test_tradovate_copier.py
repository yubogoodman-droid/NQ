#!/usr/bin/env python3
"""Unit tests for the Tradovate copier (no network)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copier.config import ConfigError, load_config, load_dotenv  # noqa: E402
from copier.engine import (  # noqa: E402
    CopyEngine,
    FollowerState,
    remap_symbol,
    session_date,
    size_qty,
)
from copier.models import (  # noqa: E402
    AccountSelector,
    ConnectionConfig,
    ConnectionCreds,
    CopierConfig,
    FollowerConfig,
)
from copier.runner import CopierApp  # noqa: E402
from copier.tradovate import (  # noqa: E402
    TradovateError,
    TradovateRest,
    decode_frame,
    encode_request,
    parse_expiration,
    resolve_account,
)

ET = ZoneInfo("America/New_York")


def make_config(**kwargs) -> CopierConfig:
    creds = ConnectionCreds("u", "p", "1", "s", "dev")
    conn = ConnectionConfig("main", "TRADOVATE", "demo", creds)
    data = dict(
        environment="demo",
        copy_mode="fills",
        dry_run=True,
        app_id="NQ Copier",
        app_version="1.0",
        flatten_when_lead_flat=True,
        flatten_on_lock=True,
        connections=[conn],
        lead=AccountSelector("main", "LEAD"),
        followers=[FollowerConfig("main", "FOLLOW", qty_ratio=1.0, max_contracts=5)],
    )
    data.update(kwargs)
    return CopierConfig(**data)


def primed_engine(config: CopierConfig | None = None) -> CopyEngine:
    engine = CopyEngine(config or make_config())
    engine.set_lead("main", 10, "LEAD")
    engine.add_follower(
        FollowerState(config=engine.config.followers[0], connection="main", account_id=20, account_spec="FOLLOW")
    )
    engine.seed_sync(
        "main",
        {
            "contracts": [{"id": 100, "name": "NQH6"}],
            "orders": [{"id": 1, "accountId": 10, "contractId": 100, "action": "Buy", "ordStatus": "Filled"}],
            "fills": [{"id": 50, "orderId": 1, "contractId": 100, "action": "Buy", "qty": 1, "active": True}],
            "positions": [
                {"accountId": 10, "contractId": 100, "netPos": 0},
                {"accountId": 20, "contractId": 100, "netPos": 0},
            ],
        },
    )
    return engine


def test_remap_symbol_does_not_treat_mnq_as_nq() -> None:
    mapping = {"NQ": "MNQ", "ES": "MES"}
    assert remap_symbol("NQH6", mapping) == "MNQH6"
    assert remap_symbol("MNQH6", mapping) == "MNQH6"
    assert remap_symbol("ESU5", mapping) == "MESU5"
    assert remap_symbol("NQ", mapping) == "MNQ"
    assert remap_symbol("CLZ6", mapping) == "CLZ6"


def test_size_qty_ratio_fixed_and_caps() -> None:
    assert size_qty(1, action="Buy", current_pos=0, qty_ratio=1.0, qty_fixed=None, max_contracts=None) == 1
    assert size_qty(1, action="Buy", current_pos=0, qty_ratio=0.4, qty_fixed=None, max_contracts=None) == 0
    assert size_qty(1, action="Buy", current_pos=0, qty_ratio=0.5, qty_fixed=None, max_contracts=None) == 1
    assert size_qty(3, action="Buy", current_pos=0, qty_ratio=1.0, qty_fixed=1, max_contracts=None) == 1
    assert size_qty(2, action="Buy", current_pos=4, qty_ratio=1.0, qty_fixed=None, max_contracts=5) == 1
    assert size_qty(2, action="Buy", current_pos=5, qty_ratio=1.0, qty_fixed=None, max_contracts=5) == 0
    assert size_qty(2, action="Sell", current_pos=5, qty_ratio=1.0, qty_fixed=None, max_contracts=5) == 2
    assert size_qty(2, action="Sell", current_pos=-5, qty_ratio=1.0, qty_fixed=None, max_contracts=5) == 0


def test_session_date_rolls_at_6pm_et() -> None:
    before = datetime(2026, 9, 8, 17, 59, tzinfo=ET)
    after = datetime(2026, 9, 8, 18, 0, tzinfo=ET)
    assert session_date(before).isoformat() == "2026-09-08"
    assert session_date(after).isoformat() == "2026-09-09"


def test_copy_lead_fill_skips_history_and_followers() -> None:
    engine = primed_engine()
    assert engine.on_fill("main", {"id": 50, "orderId": 1, "contractId": 100, "qty": 1, "action": "Buy", "active": True}) == []

    engine.orders[7] = {"id": 7, "accountId": 20, "contractId": 100, "action": "Buy"}
    assert engine.on_fill("main", {"id": 51, "orderId": 7, "contractId": 100, "qty": 1, "action": "Buy", "active": True}) == []

    engine.orders[8] = {"id": 8, "accountId": 10, "contractId": 100, "action": "Buy"}
    intents = engine.on_fill("main", {"id": 52, "orderId": 8, "contractId": 100, "qty": 2, "action": "Buy", "active": True})
    assert len(intents) == 1
    assert intents[0].kind == "place"
    assert intents[0].account_spec == "FOLLOW"
    assert intents[0].qty == 2
    assert intents[0].symbol == "NQH6"
    assert intents[0].order_type == "Market"
    assert intents[0].reason == "fill:52"


def test_pending_fill_waits_for_order_and_contract() -> None:
    engine = primed_engine()
    assert engine.on_fill("main", {"id": 80, "orderId": 9, "contractId": 200, "qty": 1, "action": "Buy", "active": True}) == []
    intents = engine.on_entity(
        "main",
        "order",
        "Created",
        {"id": 9, "accountId": 10, "contractId": 200, "action": "Buy"},
    )
    assert intents == []
    intents = engine.on_entity("main", "contract", "Created", {"id": 200, "name": "MNQH6"})
    assert len(intents) == 1
    assert intents[0].symbol == "MNQH6"


def test_symbol_map_and_own_orders_are_ignored() -> None:
    config = make_config(followers=[FollowerConfig("main", "FOLLOW", symbol_map={"NQ": "MNQ"})])
    engine = primed_engine(config)
    engine.orders[11] = {"id": 11, "accountId": 10, "contractId": 100, "action": "Sell"}
    engine.mark_own_order(11)
    assert engine.on_fill("main", {"id": 61, "orderId": 11, "contractId": 100, "qty": 1, "action": "Sell", "active": True}) == []

    engine.orders[12] = {"id": 12, "accountId": 10, "contractId": 100, "action": "Sell"}
    intents = engine.on_fill("main", {"id": 62, "orderId": 12, "contractId": 100, "qty": 1, "action": "Sell", "active": True})
    assert intents[0].symbol == "MNQH6"


def test_flatten_when_lead_goes_flat() -> None:
    engine = primed_engine()
    engine.on_entity("main", "position", "Updated", {"accountId": 10, "contractId": 100, "netPos": 2})
    engine.on_entity("main", "position", "Updated", {"accountId": 20, "contractId": 100, "netPos": 2})
    intents = engine.on_entity("main", "position", "Updated", {"accountId": 10, "contractId": 100, "netPos": 0})
    assert len(intents) == 1
    assert intents[0].kind == "flatten"
    assert intents[0].symbol == "NQH6"


def test_daily_loss_locks_and_session_unlocks() -> None:
    now = datetime(2026, 9, 8, 10, 0, tzinfo=ET)
    config = make_config(
        followers=[FollowerConfig("main", "FOLLOW", daily_loss_limit=400)],
    )
    engine = CopyEngine(config, now_fn=lambda: now)
    engine.set_lead("main", 10, "LEAD")
    engine.add_follower(FollowerState(config=config.followers[0], connection="main", account_id=20, account_spec="FOLLOW"))
    engine.contracts[100] = "NQH6"
    engine.ready = True
    engine.followers["main:20"].positions["NQH6"] = 1
    engine.followers["main:20"].contract_ids["NQH6"] = 100

    intents = engine.on_entity(
        "main",
        "cashBalance",
        "Updated",
        {"accountId": 20, "realizedPnL": -401},
    )
    assert engine.followers["main:20"].locked
    assert [item.kind for item in intents] == ["flatten"]

    engine.orders[30] = {"id": 30, "accountId": 10, "contractId": 100, "action": "Buy"}
    assert engine.on_fill("main", {"id": 90, "orderId": 30, "contractId": 100, "qty": 1, "action": "Buy", "active": True}) == []

    later = datetime(2026, 9, 8, 18, 1, tzinfo=ET)
    engine._now = lambda: later
    engine.orders[31] = {"id": 31, "accountId": 10, "contractId": 100, "action": "Buy"}
    intents = engine.on_fill("main", {"id": 91, "orderId": 31, "contractId": 100, "qty": 1, "action": "Buy", "active": True})
    assert not engine.followers["main:20"].locked
    assert len(intents) == 1


def test_orders_mode_copies_then_cancels() -> None:
    config = make_config(copy_mode="orders")
    engine = primed_engine(config)
    engine.on_entity(
        "main",
        "orderVersion",
        "Created",
        {"id": 3, "orderId": 40, "orderQty": 1, "orderType": "Limit", "price": 21000.0},
    )
    intents = engine.on_entity(
        "main",
        "order",
        "Created",
        {"id": 40, "accountId": 10, "contractId": 100, "action": "Buy", "ordStatus": "Working"},
    )
    assert len(intents) == 1
    assert intents[0].order_type == "Limit"
    assert intents[0].price == 21000.0
    engine.remember_copied_order("main:20", 40, 440)
    cancel = engine.on_entity(
        "main",
        "order",
        "Updated",
        {"id": 40, "accountId": 10, "contractId": 100, "action": "Buy", "ordStatus": "Canceled"},
    )
    assert cancel[0].kind == "cancel"
    assert cancel[0].cancel_order_id == 440

    engine.orders[41] = {"id": 41, "accountId": 10, "contractId": 100, "action": "Buy"}
    engine.copied_lead_orders.add(41)
    assert engine.on_fill("main", {"id": 99, "orderId": 41, "contractId": 100, "qty": 1, "action": "Buy", "active": True}) == []


def test_inactive_fill_is_ignored() -> None:
    engine = primed_engine()
    engine.orders[70] = {"id": 70, "accountId": 10, "contractId": 100, "action": "Buy"}
    assert engine.on_fill("main", {"id": 70, "orderId": 70, "contractId": 100, "qty": 1, "action": "Buy", "active": False}) == []
    assert 70 in engine.seen_fills


def test_ws_frames() -> None:
    assert encode_request("authorize", 2, body="tok") == "authorize\n2\n\ntok"
    assert encode_request("user/syncrequest", 3, body={"users": [9]}) == 'user/syncrequest\n3\n\n{"users":[9]}'
    assert decode_frame("o").kind == "open"
    assert decode_frame("h").kind == "heartbeat"
    frame = decode_frame('a[{"s":200,"i":2}]')
    assert frame.items[0]["s"] == 200
    props = decode_frame(
        'a[{"e":"props","d":{"entityType":"fill","eventType":"Created","entity":{"id":1}}}]'
    )
    assert props.items[0]["d"]["entityType"] == "fill"
    closed = decode_frame('c[3000,"bye"]')
    assert closed.kind == "close"


def test_parse_expiration_and_renew_skew() -> None:
    exp = parse_expiration("2026-09-06T12:00:00Z")
    assert exp is not None
    assert exp.tzinfo is not None

    calls = []

    class FakeSession:
        def request(self, method, url, headers=None, json=None, timeout=None):
            calls.append((method, url, json, headers))
            if url.endswith("/auth/accessTokenRequest"):
                return FakeResp(
                    200,
                    {
                        "accessToken": "A",
                        "userId": 7,
                        "expirationTime": "2026-09-06T12:00:00Z",
                    },
                )
            if url.endswith("/auth/renewAccessToken"):
                return FakeResp(200, {"accessToken": "B", "expirationTime": "2026-09-06T13:30:00Z"})
            if url.endswith("/account/list"):
                return FakeResp(200, [{"id": 1, "name": "LEAD"}])
            if url.endswith("/order/placeorder"):
                return FakeResp(200, {"orderId": 55})
            return FakeResp(404, {"errorText": "no"})

    now = datetime(2026, 9, 6, 11, 50, tzinfo=timezone.utc)
    rest = TradovateRest(
        "demo",
        ConnectionCreds("u", "p", "1", "s", "dev"),
        session=FakeSession(),
        now_fn=lambda: now,
    )
    rest.login()
    assert rest.access_token == "A"
    rest.ensure_token(skew_seconds=15 * 60)
    assert rest.access_token == "B"
    accounts = rest.list_accounts()
    assert accounts[0]["name"] == "LEAD"
    place = rest.place_order(accountId=1, accountSpec="LEAD", action="Buy", symbol="NQH6", orderQty=1, orderType="Market")
    assert place["orderId"] == 55
    sent = calls[-1][2]
    assert sent["isAutomated"] is True
    assert sent["text"] == "NQCOPIER"


def test_resolve_account() -> None:
    accounts = [{"id": 1, "name": "LEAD"}, {"id": 2, "name": "FOLLOW"}]
    assert resolve_account(accounts, "LEAD")["id"] == 1
    assert resolve_account(accounts, "2")["name"] == "FOLLOW"
    try:
        resolve_account(accounts, "NOPE")
        assert False, "expected error"
    except TradovateError as exc:
        assert "NOPE" in str(exc)


def test_load_config_and_reject_lead_as_follower() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        env_path = tmp_path / "copier.env"
        env_path.write_text(
            "TRADOVATE_NAME=user\nTRADOVATE_PASSWORD=secret\nTRADOVATE_CID=9\nTRADOVATE_SEC=sec\n",
            encoding="utf-8",
        )
        cfg_path = tmp_path / "copier.yaml"
        cfg_path.write_text(
            """
environment: demo
copy_mode: fills
dry_run: true
lead:
  connection: main
  account: LEAD
followers:
  - account: FOLLOW
    qty_ratio: 2
    symbol_map:
      NQ: MNQ
connections:
  - name: main
    env_prefix: TRADOVATE
""",
            encoding="utf-8",
        )
        old = {k: os.environ.pop(k, None) for k in list(os.environ) if k.startswith("TRADOVATE_")}
        try:
            config = load_config(cfg_path, env_file=env_path, device_file=tmp_path / "device")
            assert config.followers[0].qty_ratio == 2.0
            assert config.followers[0].symbol_map["NQ"] == "MNQ"
            assert config.connections[0].creds.name == "user"
            assert config.connections[0].creds.password == "secret"
            bad = tmp_path / "bad.yaml"
            bad.write_text(
                cfg_path.read_text(encoding="utf-8").replace("account: FOLLOW", "account: LEAD"),
                encoding="utf-8",
            )
            try:
                load_config(bad, env_file=env_path, device_file=tmp_path / "device2")
                assert False, "expected lead==follower error"
            except ConfigError as exc:
                assert "same account" in str(exc)
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def test_dotenv_does_not_override() -> None:
    with tempfile.TemporaryDirectory() as raw:
        path = Path(raw) / ".env"
        path.write_text("FOO=fromfile\n", encoding="utf-8")
        os.environ["FOO"] = "already"
        load_dotenv(path)
        assert os.environ["FOO"] == "already"
        del os.environ["FOO"]
        load_dotenv(path)
        assert os.environ["FOO"] == "fromfile"
        del os.environ["FOO"]


def test_runner_dry_run_and_live_execute() -> None:
    logs: list[str] = []
    fake = FakeRest()
    config = make_config(dry_run=True)
    app = CopierApp(config, rest_factory=lambda _name: fake, log=logs.append)
    app.connect_accounts()
    app.engine.ready = True
    app.engine.contracts[100] = "NQH6"
    app.engine.orders[8] = {"id": 8, "accountId": 10, "contractId": 100, "action": "Buy"}
    intents = app.handle_props(
        "main",
        {"entityType": "fill", "eventType": "Created", "entity": {"id": 52, "orderId": 8, "contractId": 100, "qty": 1, "action": "Buy", "active": True}},
    )
    assert intents and fake.placed == []
    assert any("PLACE" in line for line in logs)

    config.dry_run = False
    app.config.dry_run = False
    app.engine.orders[9] = {"id": 9, "accountId": 10, "contractId": 100, "action": "Buy"}
    app.handle_props(
        "main",
        {"entityType": "fill", "eventType": "Created", "entity": {"id": 53, "orderId": 9, "contractId": 100, "qty": 1, "action": "Buy", "active": True}},
    )
    assert fake.placed and fake.placed[0]["isAutomated"] is True
    assert fake.placed[0]["symbol"] == "NQH6"


def test_json_config() -> None:
    with tempfile.TemporaryDirectory() as raw:
        tmp_path = Path(raw)
        env_path = tmp_path / "copier.env"
        env_path.write_text(
            "TRADOVATE_NAME=user\nTRADOVATE_PASSWORD=secret\nTRADOVATE_CID=9\nTRADOVATE_SEC=sec\n",
            encoding="utf-8",
        )
        path = tmp_path / "copier.json"
        path.write_text(
            json.dumps(
                {
                    "environment": "demo",
                    "lead": {"account": "LEAD"},
                    "followers": [{"account": "FOLLOW"}],
                }
            ),
            encoding="utf-8",
        )
        old = {k: os.environ.pop(k, None) for k in list(os.environ) if k.startswith("TRADOVATE_")}
        try:
            config = load_config(path, env_file=env_path, device_file=tmp_path / "dev")
            assert config.lead.account == "LEAD"
        finally:
            for key, value in old.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


def test_engine_snapshot() -> None:
    engine = primed_engine()
    snap = engine.snapshot()
    assert snap["lead"]["name"] == "LEAD"
    assert snap["lead"]["account_id"] == 10
    assert snap["followers"][0]["name"] == "FOLLOW"
    assert snap["copy_mode"] == "fills"


def test_dashboard_page() -> None:
    html = (REPO_ROOT / "docs" / "copier" / "index.html").read_text(encoding="utf-8")
    view = (REPO_ROOT / "docs" / "copier" / "view.html").read_text(encoding="utf-8")
    for needle in ("主帳買 1 口 NQ", "symbol_map", "qty_ratio", "/api/status", "微型帳 B"):
        assert needle in html, needle
    assert "主帳買 1 口 NQ" in view


def test_demo_status() -> None:
    from copier.web import demo_status

    payload = demo_status()
    assert payload["demo"] is True
    assert payload["dry_run"] is True
    assert payload["lead"]["name"]


def test_web_cli_help() -> None:
    import subprocess

    out = subprocess.run(
        [sys.executable, str(REPO_ROOT / "examples" / "tradovate_copier.py"), "web", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert out.returncode == 0
    assert "--demo" in out.stdout


class FakeResp:
    def __init__(self, status_code: int, payload) -> None:
        self.status_code = status_code
        self._payload = payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeRest:
    def __init__(self) -> None:
        self.access_token = "t"
        self.user_id = 1
        self.placed = []
        self.liquidated = []
        self.accounts = [{"id": 10, "name": "LEAD"}, {"id": 20, "name": "FOLLOW"}]

    def login(self):
        return {"accessToken": "t", "userId": 1}

    def list_accounts(self):
        return self.accounts

    def ensure_token(self, skew_seconds: int = 0) -> None:
        return None

    def get(self, path, params=None):
        return []

    def contract_item(self, contract_id: int):
        return {"id": contract_id, "name": "NQH6"}

    def place_order(self, **fields):
        body = dict(fields)
        body["isAutomated"] = True
        self.placed.append(body)
        return {"orderId": 900 + len(self.placed)}

    def liquidate_position(self, account_id, contract_id):
        self.liquidated.append((account_id, contract_id))
        return {"orderId": 1}


def main() -> int:
    tests = [
        test_remap_symbol_does_not_treat_mnq_as_nq,
        test_size_qty_ratio_fixed_and_caps,
        test_session_date_rolls_at_6pm_et,
        test_copy_lead_fill_skips_history_and_followers,
        test_pending_fill_waits_for_order_and_contract,
        test_symbol_map_and_own_orders_are_ignored,
        test_flatten_when_lead_goes_flat,
        test_daily_loss_locks_and_session_unlocks,
        test_orders_mode_copies_then_cancels,
        test_inactive_fill_is_ignored,
        test_ws_frames,
        test_parse_expiration_and_renew_skew,
        test_resolve_account,
        test_load_config_and_reject_lead_as_follower,
        test_dotenv_does_not_override,
        test_runner_dry_run_and_live_execute,
        test_json_config,
        test_engine_snapshot,
        test_dashboard_page,
        test_demo_status,
        test_web_cli_help,
    ]
    failed = 0
    for fn in tests:
        try:
            fn()
            print(f"ok  {fn.__name__}")
        except Exception as exc:
            failed += 1
            print(f"FAIL {fn.__name__}: {exc}")
            raise
    print(f"{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
