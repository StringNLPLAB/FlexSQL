"""
mcp_launcher.py — start/stop a persistent HTTP MCP server for CCSQL.

The server reads per-request db_id / session_id from HTTP headers, so one
server process can back many sequential claude subprocesses within a fold.
Each call to `run_question` generates a temp .mcp.json pointing at this
server's URL with the right X-CCSQL-* headers baked in.

Port selection is dynamic (bind(0)) so concurrent folds on the same node
do not collide. The URL is printed to the server's stdout as a sentinel
line (`CCSQL_MCP_URL=...`) which the launcher reads to learn the port.
"""

from __future__ import annotations

import contextlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CCSQL_ROOT = Path(__file__).parent.parent.resolve()
SERVER_SCRIPT = CCSQL_ROOT / "src" / "mcp_server.py"
DEFAULT_PYTHON = CCSQL_ROOT / ".venv" / "bin" / "python"

SENTINEL = "CCSQL_MCP_URL="
READY_TIMEOUT_SEC = 30.0


@dataclass
class McpServerHandle:
    proc: subprocess.Popen
    url: str
    log_path: Path

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


def _probe_ready(url: str, timeout: float = READY_TIMEOUT_SEC) -> bool:
    """Poll the MCP URL until it responds to a streamable-http request.

    FastMCP's /mcp endpoint rejects GET without proper MCP handshake (returns
    4xx), but a response at all means the socket is up. We count any HTTP
    status response as "ready".
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                resp.read(1)
                return True
        except urllib.error.HTTPError:
            # Server responded with an HTTP error — that's fine, it's up.
            return True
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
            time.sleep(0.1)
    return False


def start_mcp_server(
    db_path: str,
    db_type: str = "snowflake",
    python_path: str | Path = DEFAULT_PYTHON,
    log_path: Path | None = None,
    tool_timeout_sec: float | None = None,
) -> McpServerHandle:
    """Spawn the MCP server in HTTP mode on a free port and wait until it's ready.

    Raises TimeoutError if the server does not come up within READY_TIMEOUT_SEC.
    """
    env = {
        **os.environ,
        "CCSQL_DB_PATH": db_path,
        "CCSQL_DB_TYPE": db_type,
    }
    if tool_timeout_sec is not None:
        env["CCSQL_TOOL_TIMEOUT_SEC"] = str(tool_timeout_sec)
    # Intentionally do NOT set CCSQL_DB_ID — HTTP requests carry it in headers.
    env.pop("CCSQL_DB_ID", None)

    cmd = [str(python_path), str(SERVER_SCRIPT), "--transport", "http", "--port", "0"]

    log_f = open(log_path, "w") if log_path else subprocess.PIPE
    proc = subprocess.Popen(
        cmd,
        cwd=str(CCSQL_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=log_f if log_path else subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Read lines from stdout until we see the sentinel (or process dies).
    url: str | None = None
    try:
        assert proc.stdout is not None
        deadline = time.time() + READY_TIMEOUT_SEC
        while time.time() < deadline:
            if proc.poll() is not None:
                break
            line = proc.stdout.readline()
            if not line:
                time.sleep(0.05)
                continue
            if log_path and log_f is not subprocess.PIPE:
                log_f.write(line)
                log_f.flush()
            line = line.strip()
            if line.startswith(SENTINEL):
                url = line[len(SENTINEL):]
                break
    except Exception:
        proc.kill()
        proc.wait()
        raise

    if url is None:
        proc.kill()
        proc.wait()
        raise TimeoutError(
            f"MCP server did not emit '{SENTINEL}...' sentinel within "
            f"{READY_TIMEOUT_SEC}s (exit={proc.returncode})"
        )

    # Drain remaining stdout into the log file (if any) in a background thread.
    if log_path and log_f is not subprocess.PIPE:
        import threading
        def _drain() -> None:
            try:
                assert proc.stdout is not None
                for chunk in proc.stdout:
                    log_f.write(chunk)
                    log_f.flush()
            except Exception:
                pass
        t = threading.Thread(target=_drain, daemon=True)
        t.start()

    if not _probe_ready(url):
        proc.kill()
        proc.wait()
        raise TimeoutError(f"MCP server at {url} did not become reachable within {READY_TIMEOUT_SEC}s")

    return McpServerHandle(proc=proc, url=url, log_path=log_path or Path("/dev/null"))


@contextlib.contextmanager
def running_mcp_server(**kwargs) -> Iterator[McpServerHandle]:
    """Context manager: start server, yield handle, always stop on exit."""
    handle = start_mcp_server(**kwargs)
    try:
        yield handle
    finally:
        handle.stop()


def write_mcp_config(
    config_path: Path,
    url: str,
    db_id: str,
    session_id: str,
    server_name: str = "snowflake-tools",
) -> Path:
    """Write a per-claude-subprocess .mcp.json pointing at the HTTP server.

    The headers tell the server which db to target and isolate the
    python_interpreter state to this claude process.
    """
    config = {
        "mcpServers": {
            server_name: {
                "type": "http",
                "url": url,
                "headers": {
                    "X-CCSQL-DB-ID": db_id,
                    "X-CCSQL-SESSION-ID": session_id,
                },
            }
        }
    }
    config_path.write_text(json.dumps(config, indent=2))
    return config_path


if __name__ == "__main__":
    # CLI convenience: start a server and block, for manual testing.
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--db-path", default=os.environ.get("CCSQL_DB_PATH", "spider2-snow"))
    p.add_argument("--db-type", default=os.environ.get("CCSQL_DB_TYPE", "snowflake"))
    args = p.parse_args()
    with running_mcp_server(db_path=args.db_path, db_type=args.db_type) as h:
        print(f"URL: {h.url}  (PID {h.proc.pid}) — Ctrl-C to stop", file=sys.stderr)
        try:
            h.proc.wait()
        except KeyboardInterrupt:
            pass
