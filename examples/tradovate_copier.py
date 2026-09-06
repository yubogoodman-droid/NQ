#!/usr/bin/env python3
"""Tradovate 本機多帳跟單（lead → followers）。

用法:
  python3 examples/tradovate_copier.py list-accounts --env-file copier.env
  python3 examples/tradovate_copier.py run --config copier.example.yaml --dry-run
  python3 examples/tradovate_copier.py run --config copier.yaml --poll 2
  python3 examples/tradovate_copier.py run --config copier.yaml --ws --live

憑證放 copier.env（勿提交）。先在 Tradovate demo 用 --dry-run 看日誌，確認無誤再拿掉 dry-run。
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from copier.config import (  # noqa: E402
    DEFAULT_ENV_FILE,
    ConfigError,
    load_config,
    load_dotenv,
    stable_device_id,
)
from copier.models import ConnectionCreds  # noqa: E402
from copier.runner import CopierApp  # noqa: E402
from copier.tradovate import TradovateError, TradovateRest  # noqa: E402


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE, help="憑證檔，預設 copier.env")
    parser.add_argument("--device-file", type=Path, default=REPO_ROOT / "copier.device_id")


def cmd_list_accounts(args: argparse.Namespace) -> int:
    load_dotenv(args.env_file)
    prefix = args.prefix
    environment = args.environment
    creds = ConnectionCreds(
        name=_need(f"{prefix}_NAME", f"{prefix}_USERNAME"),
        password=_need(f"{prefix}_PASSWORD"),
        cid=_need(f"{prefix}_CID", f"{prefix}_API_CID"),
        sec=_need(f"{prefix}_SEC", f"{prefix}_API_SECRET", f"{prefix}_SECRET"),
        device_id=_optional(f"{prefix}_DEVICE_ID") or stable_device_id(args.device_file),
        app_id="NQ Copier",
        app_version="1.0",
    )
    rest = TradovateRest(environment, creds)
    data = rest.login()
    print(f"userId={data.get('userId')} env={environment} expires={data.get('expirationTime')}")
    accounts = rest.list_accounts()
    if not accounts:
        print("no accounts")
        return 0
    width = max(len(str(item.get("name") or "")) for item in accounts)
    for item in accounts:
        print(
            f"  {str(item.get('name') or ''):<{width}}  id={item.get('id')}  "
            f"active={item.get('active')}  type={item.get('accountType')}"
        )
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    if not args.config.exists():
        raise ConfigError(f"config not found: {args.config}")
    dry_run = True if args.dry_run else None
    config = load_config(
        args.config,
        env_file=args.env_file if args.env_file.exists() else None,
        device_file=args.device_file,
        dry_run_override=dry_run,
    )
    environments = {conn.environment for conn in config.connections} | {config.environment}
    if "live" in environments and not args.live:
        raise ConfigError("live environment requires --live")
    app = CopierApp(config)
    app.log(f"mode={config.copy_mode} dry_run={config.dry_run} env={config.environment}")
    app.connect_accounts()
    if args.poll:
        app.log(f"polling every {args.poll}s")
        app.run_poll(args.poll)
    else:
        app.log("websocket user/syncrequest")
        app.run_websocket()
    return 0


def _need(*names: str) -> str:
    import os

    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    raise ConfigError("missing " + " / ".join(names))


def _optional(name: str) -> str:
    import os

    return os.environ.get(name) or ""


def main(argv: list[str] | None = None) -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="Tradovate lead/follower copier")
    sub = parser.add_subparsers(dest="cmd", required=True)

    list_p = sub.add_parser("list-accounts", help="登入並列出帳號")
    add_common(list_p)
    list_p.add_argument("--prefix", default="TRADOVATE", help="環境變數前綴，預設 TRADOVATE")
    list_p.add_argument("--environment", "--env", default="demo", choices=("demo", "live"))

    run_p = sub.add_parser("run", help="開始跟單")
    add_common(run_p)
    run_p.add_argument("--config", type=Path, default=REPO_ROOT / "copier.yaml")
    run_p.add_argument("--dry-run", action="store_true", help="只記日誌，不下單")
    run_p.add_argument("--live", action="store_true", help="允許 live 環境")
    run_p.add_argument("--poll", type=float, metavar="SEC", help="改用 REST 輪詢（秒）")
    run_p.add_argument("--ws", action="store_true", help="WebSocket（預設）")

    args = parser.parse_args(argv)
    try:
        if args.cmd == "list-accounts":
            return cmd_list_accounts(args)
        return cmd_run(args)
    except (ConfigError, TradovateError) as exc:
        logging.getLogger("copier").error(str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
