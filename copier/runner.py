from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from copier.config import CopierConfig
from copier.engine import CopyEngine, FollowerState
from copier.models import CopyIntent
from copier.tradovate import (
    TradovateError,
    TradovateRest,
    decode_frame,
    encode_request,
    resolve_account,
)

LogFn = Callable[[str], None]


def _default_log(message: str) -> None:
    logging.getLogger("copier").info(message)


class CopierApp:
    def __init__(
        self,
        config: CopierConfig,
        *,
        rest_factory: Optional[Callable[[str], TradovateRest]] = None,
        engine: Optional[CopyEngine] = None,
        log: LogFn = _default_log,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.engine = engine or CopyEngine(config)
        self.log = log
        self.sleep_fn = sleep_fn
        self.rests: Dict[str, TradovateRest] = {}
        self._lock = threading.Lock()
        environments = {conn.environment for conn in config.connections}
        if len(environments) > 1:
            raise TradovateError(
                "connections mix demo and live. Use one environment, or split into two processes."
            )
        for conn in config.connections:
            if rest_factory:
                self.rests[conn.name] = rest_factory(conn.name)
            else:
                self.rests[conn.name] = TradovateRest(conn.environment, conn.creds)

    def connect_accounts(self) -> None:
        for conn in self.config.connections:
            rest = self.rests[conn.name]
            rest.login()
            accounts = rest.list_accounts()
            self.log(
                f"{conn.name} ({conn.environment}): "
                + ", ".join(f"{a.get('name')}#{a.get('id')}" for a in accounts)
            )
            if conn.name == self.config.lead.connection:
                lead = resolve_account(accounts, self.config.lead.account)
                self.engine.set_lead(conn.name, int(lead["id"]), str(lead.get("name") or lead["id"]))
                self.log(f"lead = {lead.get('name')} id={lead.get('id')}")
            for fol in self.config.followers:
                if fol.connection != conn.name:
                    continue
                account = resolve_account(accounts, fol.account)
                state = FollowerState(
                    config=fol,
                    connection=conn.name,
                    account_id=int(account["id"]),
                    account_spec=str(account.get("name") or account["id"]),
                )
                self.engine.add_follower(state)
                self.log(
                    f"follower = {state.account_spec} id={state.account_id} "
                    f"ratio={fol.qty_ratio} map={fol.symbol_map or '-'}"
                )

        if self.engine.lead_account_id is None:
            raise TradovateError("lead account was not resolved")
        if not self.engine.followers:
            raise TradovateError("no follower accounts were resolved")

    def handle_props(self, connection: str, payload: Dict[str, Any]) -> List[CopyIntent]:
        data = payload.get("d") if "entityType" not in payload else payload
        if not isinstance(data, dict):
            return []
        entity_type = str(data.get("entityType") or "")
        event_type = str(data.get("eventType") or "")
        with self._lock:
            intents = self.engine.on_entity(connection, entity_type, event_type, data.get("entity"))
        return self.execute(intents)

    def handle_sync(self, connection: str, snapshot: Dict[str, Any]) -> List[CopyIntent]:
        contracts = list(snapshot.get("contracts") or [])
        if not contracts:
            contracts = self._fetch_contracts(
                connection,
                snapshot.get("orders"),
                snapshot.get("fills"),
                snapshot.get("positions"),
            )
        snapshot = {**snapshot, "contracts": contracts}
        with self._lock:
            intents = self.engine.seed_sync(connection, snapshot)
        self.log(f"{connection}: synced ({len(snapshot.get('fills') or [])} historical fills ignored)")
        return self.execute(intents)

    def execute(self, intents: List[CopyIntent]) -> List[CopyIntent]:
        for intent in intents:
            self.log(intent.describe())
            if self.config.dry_run:
                continue
            rest = self.rests[intent.connection]
            try:
                if intent.kind == "place":
                    result = rest.place_order(
                        accountSpec=intent.account_spec,
                        accountId=intent.account_id,
                        action=intent.action,
                        symbol=intent.symbol,
                        orderQty=intent.qty,
                        orderType=intent.order_type,
                        price=intent.price,
                        stopPrice=intent.stop_price,
                        clOrdId=f"{intent.reason}-{intent.account_id}"[:64],
                    )
                    if result.get("failureText") or result.get("failureReason"):
                        self.log(f"  REJECT {result}")
                        continue
                    order_id = result.get("orderId")
                    self.engine.mark_own_order(order_id)
                    lead_order_id = (intent.extra or {}).get("lead_order_id")
                    if lead_order_id and order_id:
                        self.engine.remember_copied_order(intent.follower_key, int(lead_order_id), int(order_id))
                    self.log(f"  ok orderId={order_id}")
                elif intent.kind == "flatten":
                    contract_id = intent.contract_id
                    if contract_id is not None:
                        result = rest.liquidate_position(intent.account_id, contract_id)
                        self.log(f"  ok {result}")
                        continue
                    follower = self.engine.followers.get(intent.follower_key)
                    net = int((follower.positions if follower else {}).get(intent.symbol or "", 0))
                    if not intent.symbol or net == 0:
                        self.log("  skip flatten: missing follower contract id")
                        continue
                    result = rest.place_order(
                        accountSpec=intent.account_spec,
                        accountId=intent.account_id,
                        action="Sell" if net > 0 else "Buy",
                        symbol=intent.symbol,
                        orderQty=abs(net),
                        orderType="Market",
                        clOrdId=f"{intent.reason}-{intent.account_id}"[:64],
                    )
                    self.engine.mark_own_order(result.get("orderId"))
                    self.log(f"  ok flatten-via-market {result}")
                elif intent.kind == "cancel" and intent.cancel_order_id:
                    result = rest.cancel_order(intent.cancel_order_id)
                    self.log(f"  ok {result}")
            except Exception as exc:
                self.log(f"  ERROR {exc}")
        return intents

    def ingest_poll_lists(
        self,
        connection: str,
        *,
        contracts: Optional[List[Dict[str, Any]]] = None,
        orders: Optional[List[Dict[str, Any]]] = None,
        fills: Optional[List[Dict[str, Any]]] = None,
        positions: Optional[List[Dict[str, Any]]] = None,
        cash: Optional[List[Dict[str, Any]]] = None,
    ) -> List[CopyIntent]:
        intents: List[CopyIntent] = []
        for contract in contracts or []:
            intents.extend(self.handle_props(connection, _props("contract", "Updated", contract)))
        for order in orders or []:
            intents.extend(self.handle_props(connection, _props("order", "Updated", order)))
        for fill in fills or []:
            intents.extend(self.handle_props(connection, _props("fill", "Created", fill)))
        for position in positions or []:
            intents.extend(self.handle_props(connection, _props("position", "Updated", position)))
        for item in cash or []:
            intents.extend(self.handle_props(connection, _props("cashBalance", "Updated", item)))
        return intents

    def poll_once(self, connection: str) -> List[CopyIntent]:
        rest = self.rests[connection]
        rest.ensure_token()
        orders = _as_list(rest.get("/order/list"))
        fills = _as_list(rest.get("/fill/list"))
        positions = _as_list(rest.get("/position/list"))
        return self.ingest_poll_lists(
            connection,
            contracts=self._fetch_contracts(connection, orders, fills, positions),
            orders=orders,
            fills=fills,
            positions=positions,
            cash=_as_list(rest.get("/cashBalance/list")),
        )

    def _fetch_contracts(self, connection: str, *groups: Any) -> List[Dict[str, Any]]:
        ids = set()
        for group in groups:
            for item in group or []:
                if isinstance(item, dict) and item.get("contractId") is not None:
                    ids.add(int(item["contractId"]))
        rest = self.rests.get(connection)
        out: List[Dict[str, Any]] = []
        for contract_id in sorted(ids):
            if contract_id in self.engine.contracts:
                continue
            if rest is None or not hasattr(rest, "contract_item"):
                continue
            try:
                item = rest.contract_item(contract_id)
            except Exception as exc:
                self.log(f"{connection}: contract {contract_id} {exc}")
                continue
            if item:
                out.append(item)
        return out

    def run_poll(self, seconds: float, stop: Optional[threading.Event] = None) -> None:
        stop = stop or threading.Event()
        first = True
        while not stop.is_set():
            for conn in self.config.connections:
                try:
                    if first:
                        self.handle_sync(
                            conn.name,
                            {
                                "orders": _as_list(self.rests[conn.name].get("/order/list")),
                                "fills": _as_list(self.rests[conn.name].get("/fill/list")),
                                "positions": _as_list(self.rests[conn.name].get("/position/list")),
                                "cashBalances": _as_list(self.rests[conn.name].get("/cashBalance/list")),
                            },
                        )
                    else:
                        self.poll_once(conn.name)
                except Exception as exc:
                    self.log(f"{conn.name} poll error: {exc}")
            first = False
            stop.wait(seconds)

    def run_websocket(self, stop: Optional[threading.Event] = None) -> None:
        try:
            import websocket  # type: ignore
        except ImportError as exc:
            raise TradovateError("websocket-client is required for --ws. pip install websocket-client") from exc

        stop = stop or threading.Event()
        threads = []
        for conn in self.config.connections:
            thread = threading.Thread(
                target=self._ws_loop,
                args=(conn.name, websocket, stop),
                name=f"ws-{conn.name}",
                daemon=True,
            )
            thread.start()
            threads.append(thread)
        try:
            while not stop.is_set():
                stop.wait(0.5)
        except KeyboardInterrupt:
            stop.set()
        for thread in threads:
            thread.join(timeout=2)

    def _ws_loop(self, connection: str, websocket_mod: Any, stop: threading.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                self._ws_once(connection, websocket_mod, stop)
                backoff = 1.0
            except Exception as exc:
                self.log(f"{connection} websocket: {exc}")
                stop.wait(backoff)
                backoff = min(30.0, backoff * 2)

    def _ws_once(self, connection: str, websocket_mod: Any, stop: threading.Event) -> None:
        rest = self.rests[connection]
        rest.ensure_token()
        if not rest.access_token or rest.user_id is None:
            rest.login()
        sock = websocket_mod.create_connection(rest.ws_url, timeout=20)
        req_id = 1
        try:
            opened = False
            authorized = False
            while not stop.is_set():
                try:
                    raw = sock.recv()
                except Exception:
                    if stop.is_set():
                        return
                    raise
                if raw is None:
                    raise TradovateError("websocket closed")
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8", "replace")
                frame = decode_frame(raw)
                if frame.kind == "open":
                    opened = True
                    sock.send(encode_request("authorize", req_id, body=rest.access_token))
                    req_id += 1
                    continue
                if frame.kind == "heartbeat":
                    sock.send("[]")
                    rest.ensure_token()
                    continue
                if frame.kind == "close":
                    raise TradovateError(f"server closed {frame.items}")
                if frame.kind != "array":
                    continue
                for item in frame.items:
                    if not isinstance(item, dict):
                        continue
                    if item.get("e") == "props":
                        self.handle_props(connection, item)
                        continue
                    if item.get("s") == 200 and not authorized and opened:
                        authorized = True
                        sock.send(
                            encode_request(
                                "user/syncrequest",
                                req_id,
                                body={"users": [rest.user_id]},
                            )
                        )
                        req_id += 1
                        continue
                    if item.get("s") not in (None, 200) and item.get("d"):
                        self.log(f"{connection} ws error {item}")
                        continue
                    data = item.get("d")
                    if isinstance(data, dict) and (
                        "accounts" in data or "orders" in data or "fills" in data or "users" in data
                    ):
                        self.handle_sync(connection, data)
        finally:
            try:
                sock.close()
            except Exception:
                pass


def _props(entity_type: str, event_type: str, entity: Dict[str, Any]) -> Dict[str, Any]:
    return {"entityType": entity_type, "eventType": event_type, "entity": entity}


def _as_list(data: Any) -> List[Dict[str, Any]]:
    if data is None:
        return []
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    if isinstance(data, dict):
        return [data]
    return []
