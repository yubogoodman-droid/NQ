from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlencode

from copier.engine import COPY_TAG
from copier.models import ConnectionCreds

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None  # type: ignore

HOSTS = {
    "demo": "https://demo.tradovateapi.com/v1",
    "live": "https://live.tradovateapi.com/v1",
}
WS_HOSTS = {
    "demo": "wss://demo.tradovateapi.com/v1/websocket",
    "live": "wss://live.tradovateapi.com/v1/websocket",
}


class TradovateError(RuntimeError):
    def __init__(self, message: str, status: Optional[int] = None, payload: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.payload = payload


@dataclass
class WsMessage:
    kind: str
    items: List[Any]
    raw: str


def encode_request(endpoint: str, req_id: int, query: str = "", body: Any = None) -> str:
    if body is None:
        payload = ""
    elif isinstance(body, str):
        payload = body
    else:
        payload = json.dumps(body, separators=(",", ":"))
    return f"{endpoint}\n{req_id}\n{query}\n{payload}"


def decode_frame(raw: str) -> WsMessage:
    if not raw:
        return WsMessage("unknown", [], raw)
    prefix, rest = raw[0], raw[1:]
    if prefix == "o":
        return WsMessage("open", [], raw)
    if prefix == "h":
        return WsMessage("heartbeat", [], raw)
    if prefix == "c":
        try:
            return WsMessage("close", json.loads(rest) if rest else [], raw)
        except json.JSONDecodeError:
            return WsMessage("close", [rest], raw)
    if prefix == "a":
        try:
            items = json.loads(rest) if rest else []
        except json.JSONDecodeError as exc:
            raise TradovateError(f"bad websocket array frame: {exc}") from exc
        if not isinstance(items, list):
            items = [items]
        return WsMessage("array", items, raw)
    return WsMessage("unknown", [], raw)


def parse_expiration(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


class TradovateRest:
    def __init__(
        self,
        environment: str,
        creds: ConnectionCreds,
        session: Any = None,
        now_fn: Optional[Callable[[], datetime]] = None,
    ) -> None:
        if environment not in HOSTS:
            raise TradovateError(f"unknown environment {environment}")
        if session is None:
            if requests is None:
                raise TradovateError("requests is required")
            session = requests.Session()
        self.environment = environment
        self.base = HOSTS[environment]
        self.ws_url = WS_HOSTS[environment]
        self.creds = creds
        self.session = session
        self.now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self.access_token: Optional[str] = None
        self.md_access_token: Optional[str] = None
        self.user_id: Optional[int] = None
        self.expiration: Optional[datetime] = None

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.access_token:
            headers["Authorization"] = f"Bearer {self.access_token}"
        return headers

    def login(self) -> Dict[str, Any]:
        body = {
            "name": self.creds.name,
            "password": self.creds.password,
            "appId": self.creds.app_id,
            "appVersion": self.creds.app_version,
            "deviceId": self.creds.device_id,
            "cid": self.creds.cid,
            "sec": self.creds.sec,
        }
        data = self._request("POST", "/auth/accessTokenRequest", json_body=body, auth=False)
        token = data.get("accessToken") if isinstance(data, dict) else None
        if not token:
            err = data.get("errorText") if isinstance(data, dict) else data
            raise TradovateError(f"login failed: {err}", payload=data)
        self._store_token(data)
        return data

    def renew(self) -> Dict[str, Any]:
        data = self._request("POST", "/auth/renewAccessToken", json_body={}, ensure=False)
        if isinstance(data, dict) and data.get("accessToken"):
            self._store_token(data)
        return data if isinstance(data, dict) else {"raw": data}

    def ensure_token(self, skew_seconds: int = 15 * 60) -> None:
        if not self.access_token:
            self.login()
            return
        if self.expiration is None:
            return
        remaining = (self.expiration - self.now_fn()).total_seconds()
        if remaining < skew_seconds:
            self.renew()

    def _store_token(self, data: Dict[str, Any]) -> None:
        self.access_token = data.get("accessToken")
        self.md_access_token = data.get("mdAccessToken")
        if data.get("userId") is not None:
            self.user_id = int(data["userId"])
        self.expiration = parse_expiration(data.get("expirationTime"))

    def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("GET", path, params=params)

    def post(self, path: str, body: Optional[Dict[str, Any]] = None) -> Any:
        return self._request("POST", path, json_body=body or {})

    def list_accounts(self) -> List[Dict[str, Any]]:
        data = self.get("/account/list")
        if data is None:
            return []
        if isinstance(data, dict):
            return [data]
        return list(data)

    def contract_item(self, contract_id: int) -> Dict[str, Any]:
        data = self.get("/contract/item", params={"id": contract_id})
        return data if isinstance(data, dict) else {}

    def place_order(self, **fields: Any) -> Dict[str, Any]:
        body = {key: value for key, value in fields.items() if value is not None}
        body["isAutomated"] = True
        body.setdefault("text", COPY_TAG)
        body.setdefault("customTag50", COPY_TAG)
        data = self.post("/order/placeorder", body)
        return data if isinstance(data, dict) else {"raw": data}

    def liquidate_position(self, account_id: int, contract_id: int) -> Dict[str, Any]:
        data = self.post(
            "/order/liquidateposition",
            {"accountId": account_id, "contractId": contract_id},
        )
        return data if isinstance(data, dict) else {"raw": data}

    def cancel_order(self, order_id: int) -> Dict[str, Any]:
        data = self.post("/order/cancelorder", {"orderId": order_id})
        return data if isinstance(data, dict) else {"raw": data}

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json_body: Optional[Dict[str, Any]] = None,
        auth: bool = True,
        ensure: bool = True,
        retried: bool = False,
    ) -> Any:
        if auth and ensure:
            self.ensure_token()
        url = self.base + path
        if params:
            url = f"{url}?{urlencode(params, doseq=True)}"
        response = self.session.request(
            method,
            url,
            headers=self._headers(),
            json=json_body if method != "GET" else None,
            timeout=30,
        )
        status = getattr(response, "status_code", None)
        try:
            data = response.json()
        except Exception:
            data = getattr(response, "text", "")
        if status == 401 and auth and not retried:
            self.login()
            return self._request(
                method,
                path,
                params=params,
                json_body=json_body,
                auth=True,
                ensure=False,
                retried=True,
            )
        if status is not None and status >= 400:
            raise TradovateError(f"{method} {path} failed ({status}): {data}", status=status, payload=data)
        if isinstance(data, dict) and data.get("errorText") and not data.get("accessToken"):
            raise TradovateError(str(data["errorText"]), status=status, payload=data)
        return data


def resolve_account(accounts: List[Dict[str, Any]], spec: str) -> Dict[str, Any]:
    wanted = str(spec).strip()
    matches = [
        item
        for item in accounts
        if str(item.get("id")) == wanted or str(item.get("name")) == wanted
    ]
    available = ", ".join(
        f"{item.get('name')} (id={item.get('id')})" for item in accounts
    ) or "(none)"
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise TradovateError(f"account {spec!r} not found. available: {available}")
    raise TradovateError(f"account {spec!r} is ambiguous. available: {available}")
