from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from copier.models import (
    AccountSelector,
    ConnectionConfig,
    ConnectionCreds,
    CopierConfig,
    FollowerConfig,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / "copier.env"
DEFAULT_DEVICE_FILE = REPO_ROOT / "copier.device_id"

VALID_ENVIRONMENTS = {"demo", "live"}
VALID_MODES = {"fills", "orders"}


class ConfigError(ValueError):
    pass


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def env_get(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.environ.get(name, default)
    return value if value not in (None, "") else default


def stable_device_id(path: Path = DEFAULT_DEVICE_FILE) -> str:
    if path.exists():
        existing = path.read_text(encoding="utf-8").strip()
        if existing:
            return existing
    device_id = str(uuid.uuid4())
    path.write_text(device_id + "\n", encoding="utf-8")
    return device_id


def _read_config_file(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ConfigError("YAML config needs PyYAML. Use a .json file or pip install pyyaml.") from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ConfigError("config root must be a mapping")
    return data


def _require_str(data: Dict[str, Any], key: str, default: Optional[str] = None) -> str:
    value = data.get(key, default)
    if value is None or str(value).strip() == "":
        raise ConfigError(f"missing {key}")
    return str(value).strip()


def _creds_from_env(prefix: str, app_id: str, app_version: str, device_id: str) -> ConnectionCreds:
    names = [f"{prefix}_NAME", f"{prefix}_USERNAME"]
    username = next((env_get(name) for name in names if env_get(name)), None)
    password = env_get(f"{prefix}_PASSWORD")
    cid = env_get(f"{prefix}_CID") or env_get(f"{prefix}_API_CID")
    sec = env_get(f"{prefix}_SEC") or env_get(f"{prefix}_API_SECRET") or env_get(f"{prefix}_SECRET")
    device = env_get(f"{prefix}_DEVICE_ID") or device_id
    missing = [label for label, value in (("NAME", username), ("PASSWORD", password), ("CID", cid), ("SEC", sec)) if not value]
    if missing:
        raise ConfigError(
            f"missing env for {prefix}: {', '.join(prefix + '_' + item for item in missing)}"
        )
    return ConnectionCreds(
        name=username or "",
        password=password or "",
        cid=str(cid),
        sec=str(sec),
        device_id=device,
        app_id=app_id,
        app_version=app_version,
    )


def _parse_follower(raw: Dict[str, Any], default_connection: str) -> FollowerConfig:
    account = _require_str(raw, "account")
    qty_fixed = raw.get("qty_fixed")
    max_contracts = raw.get("max_contracts")
    daily_loss = raw.get("daily_loss_limit")
    symbol_map = raw.get("symbol_map") or {}
    if not isinstance(symbol_map, dict):
        raise ConfigError("follower.symbol_map must be a mapping")
    ratio = float(raw.get("qty_ratio", 1.0))
    if ratio < 0:
        raise ConfigError("qty_ratio must be >= 0")
    return FollowerConfig(
        connection=str(raw.get("connection") or default_connection),
        account=account,
        qty_ratio=ratio,
        qty_fixed=int(qty_fixed) if qty_fixed is not None else None,
        symbol_map={str(k).upper(): str(v).upper() for k, v in symbol_map.items()},
        max_contracts=int(max_contracts) if max_contracts is not None else None,
        daily_loss_limit=float(daily_loss) if daily_loss is not None else None,
        enabled=bool(raw.get("enabled", True)),
    )


def load_config(
    path: Path,
    *,
    env_file: Optional[Path] = None,
    device_file: Path = DEFAULT_DEVICE_FILE,
    dry_run_override: Optional[bool] = None,
) -> CopierConfig:
    if env_file:
        load_dotenv(env_file)
    elif DEFAULT_ENV_FILE.exists():
        load_dotenv(DEFAULT_ENV_FILE)

    raw = _read_config_file(path)
    environment = str(raw.get("environment") or "demo").strip().lower()
    if environment not in VALID_ENVIRONMENTS:
        raise ConfigError(f"environment must be demo or live, got {environment}")

    copy_mode = str(raw.get("copy_mode") or "fills").strip().lower()
    if copy_mode not in VALID_MODES:
        raise ConfigError(f"copy_mode must be fills or orders, got {copy_mode}")

    app_id = str(raw.get("app_id") or "NQ Copier")
    app_version = str(raw.get("app_version") or "1.0")
    risk = raw.get("risk") or {}
    if not isinstance(risk, dict):
        raise ConfigError("risk must be a mapping")

    connections_raw = raw.get("connections")
    if not connections_raw:
        connections_raw = [{"name": "main", "env_prefix": "TRADOVATE", "environment": environment}]
    if not isinstance(connections_raw, list) or not connections_raw:
        raise ConfigError("connections must be a non-empty list")

    device_id = stable_device_id(device_file)
    connections: List[ConnectionConfig] = []
    for item in connections_raw:
        if not isinstance(item, dict):
            raise ConfigError("each connection must be a mapping")
        name = _require_str(item, "name")
        prefix = str(item.get("env_prefix") or f"TRADOVATE_{name.upper()}")
        conn_env = str(item.get("environment") or environment).strip().lower()
        if conn_env not in VALID_ENVIRONMENTS:
            raise ConfigError(f"connection {name}: environment must be demo or live")
        connections.append(
            ConnectionConfig(
                name=name,
                env_prefix=prefix,
                environment=conn_env,
                creds=_creds_from_env(prefix, app_id, app_version, device_id),
            )
        )

    lead_raw = raw.get("lead")
    if not isinstance(lead_raw, dict):
        raise ConfigError("lead must be a mapping with connection + account")
    default_conn = connections[0].name
    lead = AccountSelector(
        connection=str(lead_raw.get("connection") or default_conn),
        account=_require_str(lead_raw, "account"),
    )

    followers_raw = raw.get("followers")
    if not isinstance(followers_raw, list) or not followers_raw:
        raise ConfigError("followers must be a non-empty list")
    followers: List[FollowerConfig] = []
    for item in followers_raw:
        if not isinstance(item, dict):
            raise ConfigError("each follower must be a mapping")
        followers.append(_parse_follower(item, default_conn))

    conn_names = {c.name for c in connections}
    for selector in [lead, *followers]:
        if selector.connection not in conn_names:
            raise ConfigError(f"unknown connection {selector.connection!r}")

    lead_key = (lead.connection, str(lead.account))
    for fol in followers:
        if (fol.connection, str(fol.account)) == lead_key:
            raise ConfigError("a follower cannot be the same account as lead")

    dry_run = bool(raw.get("dry_run", True))
    if dry_run_override is not None:
        dry_run = dry_run_override

    return CopierConfig(
        environment=environment,
        copy_mode=copy_mode,
        dry_run=dry_run,
        app_id=app_id,
        app_version=app_version,
        flatten_when_lead_flat=bool(risk.get("flatten_followers_when_lead_flat", True)),
        flatten_on_lock=bool(risk.get("flatten_on_lock", True)),
        connections=connections,
        lead=lead,
        followers=followers,
    )
