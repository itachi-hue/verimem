#!/usr/bin/env python3
"""
Minimal Hindsight API smoke test (retain → recall → delete bank).

Requires a running Hindsight server (Docker is the supported path on Windows).
``hindsight-all`` embedded mode is not supported on Windows (uvloop).

Usage::

    pip install hindsight-client
    python benchmarks/hindsight_smoke.py
    python benchmarks/hindsight_smoke.py --url http://127.0.0.1:8888

Exit 0 on success, 1 on failure.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys
import uuid
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
_BENCH = Path(__file__).resolve().parent
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))


def _tcp_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def main() -> int:
    p = argparse.ArgumentParser(description="Hindsight API smoke (retain + recall)")
    p.add_argument(
        "--url",
        default="http://127.0.0.1:8888",
        help="Hindsight API base URL",
    )
    p.add_argument("--timeout", type=float, default=120.0, help="HTTP client timeout (s)")
    args = p.parse_args()

    base = args.url.rstrip("/")
    parsed = urlparse(base)
    host = parsed.hostname or "127.0.0.1"
    # Docker Desktop often sets HTTP_PROXY; aiohttp must not send loopback via the proxy.
    if host in ("127.0.0.1", "localhost", "::1"):
        for key in ("NO_PROXY", "no_proxy"):
            extra = "127.0.0.1,localhost"
            cur = (os.environ.get(key) or "").strip()
            if extra not in cur:
                os.environ[key] = f"{extra},{cur}".strip(",")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if port == 80 and ":8888" in base:
        port = 8888

    print(f"Hindsight smoke: {base}", flush=True)
    if not _tcp_open(host, port):
        print(
            f"ERROR: Nothing listening on {host}:{port}.\n"
            "On Windows, run the official container (Docker Desktop must be installed and running), e.g.:\n"
            "  docker run -d --name hindsight -p 8888:8888 -p 9999:9999 ^\n"
            "    -e HINDSIGHT_API_LLM_API_KEY=%GROQ_OR_OPENAI_KEY% ^\n"
            "    -e HINDSIGHT_API_LLM_PROVIDER=groq ^\n"
            r"    -v %USERPROFILE%\.hindsight-docker:/home/hindsight/.pg0 ^"
            "\n    ghcr.io/vectorize-io/hindsight:latest\n"
            "Then set the same key provider the container expects for retain/extract.\n"
            "Docs: https://github.com/vectorize-io/hindsight",
            file=sys.stderr,
        )
        return 1

    try:
        from hindsight_client import Hindsight
    except ImportError:
        print("ERROR: pip install hindsight-client", file=sys.stderr)
        return 1

    bank_id = f"smoke_{uuid.uuid4().hex[:20]}"
    client = Hindsight(base_url=base, api_key=None, timeout=args.timeout)

    try:
        print(f"  create_bank({bank_id!r})...")
        client.create_bank(bank_id=bank_id)
        try:
            print("  retain('Alice works at Acme as an engineer.')...")
            client.retain(
                bank_id=bank_id,
                content="Alice works at Acme as an engineer.",
                document_id="0",
            )
            print("  recall('Where does Alice work?')...")
            resp = client.recall(bank_id=bank_id, query="Where does Alice work?", max_tokens=2048)
            n = len(resp.results or [])
            print(f"  -> {n} recall result(s)")
            if n == 0:
                print(
                    "WARNING: recall returned no results (server up but pipeline may need LLM quota/model).",
                    file=sys.stderr,
                )
        finally:
            print(f"  delete_bank({bank_id!r})...")
            try:
                client.delete_bank(bank_id=bank_id)
            except Exception as e:
                print(f"WARNING: delete_bank failed: {e}", file=sys.stderr)
    finally:
        try:
            client.close()
        except Exception:
            pass

    print("OK - Hindsight API smoke passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
