from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ConnectionCreds:
    name: str
    password: str
    cid: str
    sec: str
    device_id: str
    app_id: str = "NQ Copier"
    app_version: str = "1.0"


@dataclass
class ConnectionConfig:
    name: str
    env_prefix: str
    environment: str
    creds: ConnectionCreds


@dataclass
class AccountSelector:
    connection: str
    account: str  # Tradovate account name or numeric id


@dataclass
class FollowerConfig:
    connection: str
    account: str
    qty_ratio: float = 1.0
    qty_fixed: Optional[int] = None
    symbol_map: Dict[str, str] = field(default_factory=dict)
    max_contracts: Optional[int] = None
    daily_loss_limit: Optional[float] = None
    enabled: bool = True


@dataclass
class CopierConfig:
    environment: str
    copy_mode: str  # fills | orders
    dry_run: bool
    app_id: str
    app_version: str
    flatten_when_lead_flat: bool
    flatten_on_lock: bool
    connections: List[ConnectionConfig]
    lead: AccountSelector
    followers: List[FollowerConfig]


@dataclass
class CopyIntent:
    kind: str  # place | flatten | cancel
    follower_key: str
    connection: str
    account_id: int
    account_spec: str
    reason: str
    action: Optional[str] = None
    symbol: Optional[str] = None
    contract_id: Optional[int] = None
    qty: Optional[int] = None
    order_type: str = "Market"
    price: Optional[float] = None
    stop_price: Optional[float] = None
    cancel_order_id: Optional[int] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def describe(self) -> str:
        if self.kind == "flatten":
            target = self.symbol or (f"contract:{self.contract_id}" if self.contract_id else "?")
            return f"FLATTEN {self.account_spec} {target} ({self.reason})"
        if self.kind == "cancel":
            return f"CANCEL {self.account_spec} order {self.cancel_order_id} ({self.reason})"
        return (
            f"PLACE {self.account_spec} {self.action} {self.qty} {self.symbol} "
            f"{self.order_type} ({self.reason})"
        )
