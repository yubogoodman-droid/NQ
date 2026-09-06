from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from zoneinfo import ZoneInfo

from copier.models import CopierConfig, CopyIntent, FollowerConfig

ET = ZoneInfo("America/New_York")
COPY_TAG = "NQCOPIER"
CONTRACT_RE = re.compile(r"^([A-Z]+)([FGHJKMNQUVXZ])(\d{1,2})$")
WORKING = {"Working", "PendingNew", "Accepted", "Suspended"}
DEAD = {"Canceled", "Cancelled", "Expired", "Rejected", "Unknown"}


def session_date(now: datetime) -> datetime.date:
    """Futures session date: rolls at 18:00 America/New_York."""
    et = now.astimezone(ET) if now.tzinfo else now.replace(tzinfo=ET)
    if et.hour >= 18:
        return (et + timedelta(days=1)).date()
    return et.date()


def split_contract(symbol: str) -> Tuple[str, str]:
    text = (symbol or "").strip().upper()
    match = CONTRACT_RE.match(text)
    if match:
        return match.group(1), match.group(2) + match.group(3)
    return text, ""


def remap_symbol(symbol: str, mapping: Dict[str, str]) -> str:
    if not symbol:
        return symbol
    product, suffix = split_contract(symbol)
    if not mapping:
        return symbol.upper() if suffix else symbol
    keys = sorted(mapping, key=len, reverse=True)
    for key in keys:
        if product == key.upper():
            dest = mapping[key]
            return f"{dest}{suffix}" if suffix else dest
    return symbol.upper() if suffix else symbol


def size_qty(
    lead_qty: int,
    *,
    action: str,
    current_pos: int,
    qty_ratio: float,
    qty_fixed: Optional[int],
    max_contracts: Optional[int],
) -> int:
    if lead_qty <= 0:
        return 0
    if qty_fixed is not None:
        qty = max(0, int(qty_fixed))
    else:
        qty = int(round(lead_qty * qty_ratio))
        qty = max(0, qty)
    if qty <= 0:
        return 0
    if max_contracts is None:
        return qty
    cap = int(max_contracts)
    if action == "Buy":
        allowed = cap - current_pos
    else:
        allowed = cap + current_pos
    return max(0, min(qty, allowed))


def iter_entities(entity: Any) -> Iterable[Dict[str, Any]]:
    if entity is None:
        return
    if isinstance(entity, list):
        for item in entity:
            if isinstance(item, dict):
                yield item
        return
    if isinstance(entity, dict):
        yield entity


@dataclass
class FollowerState:
    config: FollowerConfig
    connection: str
    account_id: int
    account_spec: str
    locked: bool = False
    lock_reason: str = ""
    realized_pnl: float = 0.0
    positions: Dict[str, int] = field(default_factory=dict)
    contract_ids: Dict[str, int] = field(default_factory=dict)
    copied_from_lead: Dict[int, int] = field(default_factory=dict)


class CopyEngine:
    """Pure lead/follower copy rules. No network I/O."""

    def __init__(
        self,
        config: CopierConfig,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config
        self._now = now_fn or (lambda: datetime.now(ET))
        self.lead_connection: Optional[str] = None
        self.lead_account_id: Optional[int] = None
        self.lead_account_spec: str = ""
        self.followers: Dict[str, FollowerState] = {}
        self.orders: Dict[int, Dict[str, Any]] = {}
        self.order_versions: Dict[int, Dict[str, Any]] = {}
        self.contracts: Dict[int, str] = {}
        self.seen_fills: set[int] = set()
        self.pending_fills: Dict[int, List[Dict[str, Any]]] = {}
        self.pending_no_symbol: List[Tuple[str, Dict[str, Any]]] = []
        self.own_order_ids: set[int] = set()
        self.copied_lead_orders: set[int] = set()
        self.lead_positions: Dict[str, int] = {}
        self.lead_contract_ids: Dict[str, int] = {}
        self._session = session_date(self._now())
        self.ready = False

    @staticmethod
    def follower_key(connection: str, account_id: int) -> str:
        return f"{connection}:{account_id}"

    def set_lead(self, connection: str, account_id: int, account_spec: str) -> None:
        self.lead_connection = connection
        self.lead_account_id = int(account_id)
        self.lead_account_spec = account_spec

    def add_follower(self, state: FollowerState) -> None:
        self.followers[self.follower_key(state.connection, state.account_id)] = state

    def mark_own_order(self, order_id: Optional[int]) -> None:
        if order_id:
            self.own_order_ids.add(int(order_id))

    def seed_sync(self, connection: str, snapshot: Dict[str, Any]) -> List[CopyIntent]:
        for contract in snapshot.get("contracts") or []:
            self._remember_contract(contract)
        for order in snapshot.get("orders") or []:
            self.orders[int(order["id"])] = order
            if (
                self.lead_account_id is not None
                and int(order.get("accountId") or 0) == self.lead_account_id
                and connection == self.lead_connection
            ):
                self.copied_lead_orders.add(int(order["id"]))
        for version in snapshot.get("orderVersions") or []:
            self._put_version(version)
        for fill in snapshot.get("fills") or []:
            if "id" in fill:
                self.seen_fills.add(int(fill["id"]))
        intents: List[CopyIntent] = []
        for position in snapshot.get("positions") or []:
            intents.extend(self._apply_position(connection, position, emit=False))
        for cash in snapshot.get("cashBalances") or []:
            intents.extend(self._apply_cash(connection, cash, flatten=False))
        self.ready = True
        return intents

    def on_entity(
        self,
        connection: str,
        entity_type: str,
        event_type: str,
        entity: Any,
    ) -> List[CopyIntent]:
        self._maybe_roll_session()
        intents: List[CopyIntent] = []
        kind = (entity_type or "").lower()
        for item in iter_entities(entity):
            if kind == "contract":
                self._remember_contract(item)
                intents.extend(self._flush_pending_no_symbol())
            elif kind == "order":
                intents.extend(self._on_order(connection, item))
            elif kind == "orderversion":
                self._put_version(item)
            elif kind == "fill":
                intents.extend(self.on_fill(connection, item))
            elif kind == "position":
                intents.extend(self._apply_position(connection, item, emit=self.ready))
            elif kind == "cashbalance":
                intents.extend(self._apply_cash(connection, item, flatten=self.ready))
        return intents

    def on_fill(self, connection: str, fill: Dict[str, Any]) -> List[CopyIntent]:
        if "id" not in fill:
            return []
        fill_id = int(fill["id"])
        if fill_id in self.seen_fills:
            return []
        if not fill.get("active", True):
            self.seen_fills.add(fill_id)
            return []

        order_id = fill.get("orderId")
        if order_id is None:
            self.seen_fills.add(fill_id)
            return []
        order_id = int(order_id)
        order = self.orders.get(order_id)
        if order is None:
            bucket = self.pending_fills.setdefault(order_id, [])
            if not any(int(item.get("id") or 0) == fill_id for item in bucket):
                bucket.append(fill)
            return []

        symbol = self._symbol_for(fill.get("contractId") or order.get("contractId"))
        if not symbol:
            if not any(int(item.get("id") or 0) == fill_id for _, item in self.pending_no_symbol):
                self.pending_no_symbol.append((connection, fill))
            return []

        self.seen_fills.add(fill_id)
        if not self.ready:
            return []
        if order_id in self.own_order_ids:
            return []
        if int(order.get("accountId") or 0) != self.lead_account_id:
            return []
        if connection != self.lead_connection:
            return []
        if self.config.copy_mode == "orders" and order_id in self.copied_lead_orders:
            return []

        action = str(fill.get("action") or order.get("action") or "")
        qty = int(fill.get("qty") or 0)
        if action not in {"Buy", "Sell"} or qty <= 0:
            return []
        return self._intents_for_fill(fill_id, action, qty, symbol)

    def _intents_for_fill(self, fill_id: int, action: str, qty: int, symbol: str) -> List[CopyIntent]:
        intents: List[CopyIntent] = []
        for key, follower in self.followers.items():
            if not follower.config.enabled or follower.locked:
                continue
            dest = remap_symbol(symbol, follower.config.symbol_map)
            sized = size_qty(
                qty,
                action=action,
                current_pos=int(follower.positions.get(dest, 0)),
                qty_ratio=follower.config.qty_ratio,
                qty_fixed=follower.config.qty_fixed,
                max_contracts=follower.config.max_contracts,
            )
            if sized <= 0:
                continue
            intents.append(
                CopyIntent(
                    kind="place",
                    follower_key=key,
                    connection=follower.connection,
                    account_id=follower.account_id,
                    account_spec=follower.account_spec,
                    action=action,
                    symbol=dest,
                    contract_id=follower.contract_ids.get(dest),
                    qty=sized,
                    order_type="Market",
                    reason=f"fill:{fill_id}",
                    extra={"lead_symbol": symbol},
                )
            )
        return intents

    def _on_order(self, connection: str, order: Dict[str, Any]) -> List[CopyIntent]:
        if "id" not in order:
            return []
        order_id = int(order["id"])
        merged = {**self.orders.get(order_id, {}), **order}
        self.orders[order_id] = merged
        intents: List[CopyIntent] = []
        for fill in self.pending_fills.pop(order_id, []):
            intents.extend(self.on_fill(connection, fill))
        if self.config.copy_mode == "orders" and self.ready:
            intents.extend(self._copy_working_order(connection, merged))
        return intents

    def _copy_working_order(self, connection: str, order: Dict[str, Any]) -> List[CopyIntent]:
        order_id = int(order["id"])
        if order_id in self.own_order_ids:
            return []
        if int(order.get("accountId") or 0) != self.lead_account_id:
            return []
        if connection != self.lead_connection:
            return []
        if order.get("admin"):
            return []
        status = str(order.get("ordStatus") or "")
        if status in DEAD and order_id in self.copied_lead_orders:
            return self._cancel_copied(order_id, f"lead-order:{order_id}:{status}")
        if status not in WORKING or order_id in self.copied_lead_orders:
            return []
        version = self.order_versions.get(order_id)
        if not version:
            return []
        symbol = self._symbol_for(order.get("contractId"))
        if not symbol:
            return []
        action = str(order.get("action") or "")
        qty = int(version.get("orderQty") or 0)
        order_type = str(version.get("orderType") or "Market")
        if action not in {"Buy", "Sell"} or qty <= 0:
            return []
        intents: List[CopyIntent] = []
        for key, follower in self.followers.items():
            if not follower.config.enabled or follower.locked:
                continue
            dest = remap_symbol(symbol, follower.config.symbol_map)
            sized = size_qty(
                qty,
                action=action,
                current_pos=int(follower.positions.get(dest, 0)),
                qty_ratio=follower.config.qty_ratio,
                qty_fixed=follower.config.qty_fixed,
                max_contracts=follower.config.max_contracts,
            )
            if sized <= 0:
                continue
            intents.append(
                CopyIntent(
                    kind="place",
                    follower_key=key,
                    connection=follower.connection,
                    account_id=follower.account_id,
                    account_spec=follower.account_spec,
                    action=action,
                    symbol=dest,
                    qty=sized,
                    order_type=order_type,
                    price=version.get("price"),
                    stop_price=version.get("stopPrice"),
                    reason=f"order:{order_id}",
                    extra={"lead_order_id": order_id},
                )
            )
        if intents:
            self.copied_lead_orders.add(order_id)
        return intents

    def _cancel_copied(self, lead_order_id: int, reason: str) -> List[CopyIntent]:
        intents: List[CopyIntent] = []
        for key, follower in self.followers.items():
            mapped = follower.copied_from_lead.get(lead_order_id)
            if not mapped:
                continue
            intents.append(
                CopyIntent(
                    kind="cancel",
                    follower_key=key,
                    connection=follower.connection,
                    account_id=follower.account_id,
                    account_spec=follower.account_spec,
                    cancel_order_id=mapped,
                    reason=reason,
                )
            )
        return intents

    def remember_copied_order(self, follower_key: str, lead_order_id: int, follower_order_id: int) -> None:
        follower = self.followers.get(follower_key)
        if follower:
            follower.copied_from_lead[lead_order_id] = follower_order_id
            self.mark_own_order(follower_order_id)

    def _apply_position(self, connection: str, position: Dict[str, Any], *, emit: bool) -> List[CopyIntent]:
        account_id = position.get("accountId")
        contract_id = position.get("contractId")
        if account_id is None or contract_id is None:
            return []
        account_id = int(account_id)
        contract_id = int(contract_id)
        symbol = self._symbol_for(contract_id)
        if not symbol:
            return []
        net = int(position.get("netPos") or 0)
        if account_id == self.lead_account_id and connection == self.lead_connection:
            prev = self.lead_positions.get(symbol)
            self.lead_positions[symbol] = net
            self.lead_contract_ids[symbol] = contract_id
            if emit and self.config.flatten_when_lead_flat and prev not in (None, 0) and net == 0:
                return self._flatten_followers(symbol, f"lead-flat:{symbol}")
            return []
        key = self.follower_key(connection, account_id)
        follower = self.followers.get(key)
        if follower:
            follower.positions[symbol] = net
            follower.contract_ids[symbol] = contract_id
        return []

    def _flatten_followers(self, lead_symbol: str, reason: str) -> List[CopyIntent]:
        intents: List[CopyIntent] = []
        for key, follower in self.followers.items():
            if not follower.config.enabled:
                continue
            dest = remap_symbol(lead_symbol, follower.config.symbol_map)
            net = int(follower.positions.get(dest, 0))
            if net == 0:
                continue
            intents.append(
                CopyIntent(
                    kind="flatten",
                    follower_key=key,
                    connection=follower.connection,
                    account_id=follower.account_id,
                    account_spec=follower.account_spec,
                    symbol=dest,
                    contract_id=follower.contract_ids.get(dest),
                    reason=reason,
                )
            )
        return intents

    def _apply_cash(self, connection: str, cash: Dict[str, Any], *, flatten: bool) -> List[CopyIntent]:
        account_id = cash.get("accountId")
        if account_id is None:
            return []
        key = self.follower_key(connection, int(account_id))
        follower = self.followers.get(key)
        if not follower or follower.config.daily_loss_limit is None:
            return []
        pnl = float(cash.get("realizedPnL") or 0.0)
        follower.realized_pnl = pnl
        limit = abs(float(follower.config.daily_loss_limit))
        if pnl > -limit:
            return []
        if follower.locked:
            return []
        follower.locked = True
        follower.lock_reason = f"daily_loss:{pnl}"
        if not flatten or not self.config.flatten_on_lock:
            return []
        intents: List[CopyIntent] = []
        for symbol, net in list(follower.positions.items()):
            if net == 0:
                continue
            intents.append(
                CopyIntent(
                    kind="flatten",
                    follower_key=key,
                    connection=follower.connection,
                    account_id=follower.account_id,
                    account_spec=follower.account_spec,
                    symbol=symbol,
                    contract_id=follower.contract_ids.get(symbol),
                    reason=follower.lock_reason,
                )
            )
        return intents

    def _flush_pending_no_symbol(self) -> List[CopyIntent]:
        leftover: List[Tuple[str, Dict[str, Any]]] = []
        intents: List[CopyIntent] = []
        for connection, fill in self.pending_no_symbol:
            symbol = self._symbol_for(fill.get("contractId"))
            if not symbol:
                leftover.append((connection, fill))
                continue
            intents.extend(self.on_fill(connection, fill))
        self.pending_no_symbol = leftover
        return intents

    def _remember_contract(self, contract: Dict[str, Any]) -> None:
        if "id" in contract and contract.get("name"):
            self.contracts[int(contract["id"])] = str(contract["name"])

    def _put_version(self, version: Dict[str, Any]) -> None:
        order_id = version.get("orderId")
        if order_id is None:
            return
        order_id = int(order_id)
        current = self.order_versions.get(order_id)
        if current is None or int(version.get("id") or 0) >= int(current.get("id") or 0):
            self.order_versions[order_id] = version

    def _symbol_for(self, contract_id: Any) -> str:
        if contract_id is None:
            return ""
        return self.contracts.get(int(contract_id), "")

    def _maybe_roll_session(self) -> None:
        today = session_date(self._now())
        if today == self._session:
            return
        self._session = today
        for follower in self.followers.values():
            follower.locked = False
            follower.lock_reason = ""
            follower.realized_pnl = 0.0
