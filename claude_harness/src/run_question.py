#!/usr/bin/env python3
"""
run_question.py — single-shot Claude Code agent for text-to-SQL.

Runs one `claude -p` invocation per question. The agent has access to the
MCP tools declared in .mcp.json, the hard rules in CLAUDE.md, and the
high-level guidance in workflow.md (appended to the system prompt).

The agent is free to plan, probe, write, and revise in whatever order it
chooses — it just has to write a single SQL query to `answer.sql` before
the wall-clock budget expires.

Output per question in {output_dir}/{instance_id}/:
  - answer.sql            — final SQL the agent wrote
  - answer_formatted.sql  — post-formatted SQL (quoted identifiers)
  - {instance_id}.csv     — query execution result
  - pipeline.json         — harness metadata (elapsed, status, exit code)
  - logs/
      raw.jsonl           — raw Claude Code stream-json
      stderr.log          — subprocess stderr
      log.txt             — human-readable parsed transcript
"""

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from mcp_launcher import running_mcp_server, write_mcp_config

CCSQL_ROOT = Path(__file__).parent.parent.resolve()
WORKFLOW_PATH = CCSQL_ROOT / "workflow.md"


# ── Prompt building ───────────────────────────────────────────────────────────


def _knowledge_section(question: dict, db_path: str) -> str:
    """Optional external-knowledge hint block for questions that reference docs."""
    external_knowledge = question.get("external_knowledge") or ""
    if not external_knowledge.strip():
        return ""
    doc_path = os.path.join(os.path.abspath(db_path), "resource", "documents", external_knowledge)
    if os.path.exists(doc_path):
        return (
            f"\nExternal documentation is available at: {doc_path}\n"
            f"Read this file if you need domain-specific knowledge to answer the question.\n"
        )
    return f"\nExternal knowledge hint: {external_knowledge}\n"


def build_prompt(question: dict, db_path: str, answer_file: Path) -> str:
    instruction = question.get("instruction", "")
    db_id = question.get("db_id", "")
    knowledge = _knowledge_section(question, db_path)
    return (
        f"You are answering a single text-to-SQL question on the Snowflake database `{db_id}`.\n\n"
        f"Question:\n{instruction}\n"
        f"{knowledge}\n"
        f"Treat this as an iterative investigation, not a pipeline. You have MCP tools for database exploration "
        f"and two skills (`sql-writing`, `sql-revision`) loadable via the Skill tool — these are resources you "
        f"can invoke at any time, in any order, more than once. `sql-writing` is the upstream investigation "
        f"(explore, identify ambiguity, test interpretations against data, commit, produce SQL). `sql-revision` "
        f"is the skeptical downstream check (verify the output literally answers the question).\n\n"
        f"You are **not** done when you have written SQL that runs. You are done when you have verified the "
        f"answer by running queries that try to break it — alternate aggregations, different filter values, "
        f"counts that should match, plausibility checks on the output — and cannot find a mistake. Probe as "
        f"aggressively during verification as during exploration.\n\n"
        f"See `workflow.md` for how to approach the task. Consult `CLAUDE.md` for tool signatures and SQL rules.\n\n"
        f"Write your final SQL query to: {answer_file}\n"
        f"Raw SQL only — no markdown fences, no comments, no explanation. One query. "
        f"The harness reads this file as your answer; if it is missing or malformed, the question is scored as failed."
    )


# ── Claude subprocess runner ──────────────────────────────────────────────────


def _run_claude(
    prompt: str,
    system_append: str,
    env: dict,
    logs_dir: Path,
    timeout: int,
    claude_path: str,
    model: str,
    effort: str | None = None,
    mcp_config_path: Path | None = None,
) -> dict:
    raw_log = logs_dir / "raw.jsonl"
    stderr_path = logs_dir / "stderr.log"
    log_file = logs_dir / "log.txt"

    cmd = [
        claude_path, "-p", prompt,
        "--output-format", "stream-json", "--verbose",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--disable-slash-commands",
        "--append-system-prompt", system_append,
    ]
    if mcp_config_path is not None:
        cmd += ["--mcp-config", str(mcp_config_path), "--strict-mcp-config"]
    if effort:
        cmd += ["--effort", effort]

    # Record the exact CLI invocation so --effort / --model are auditable
    # post-hoc. Prompt and system-append are large; redact them to keep the
    # file scannable.
    redacted = [
        "<PROMPT>" if a is prompt else "<SYSTEM_APPEND>" if a is system_append else a
        for a in cmd
    ]
    (logs_dir / "cmd.txt").write_text(" ".join(redacted) + "\n")

    t0 = time.time()
    proc = None
    timed_out = False
    try:
        with open(raw_log, "w") as raw_f, open(stderr_path, "w") as err_f:
            proc = subprocess.Popen(
                cmd, env=env, cwd=str(CCSQL_ROOT),
                stdout=raw_f, stderr=err_f, text=True,
            )
            returncode = proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        returncode = -1
        if proc:
            proc.kill()
            proc.wait()
    except Exception:
        if proc and proc.poll() is None:
            proc.kill()
            proc.wait()
        raise
    elapsed = time.time() - t0

    raw_lines = raw_log.read_text().strip().splitlines() if raw_log.exists() else []
    readable = parse_stream_to_log(raw_lines)
    header = (
        f"Elapsed: {elapsed:.1f}s{'  (TIMEOUT)' if timed_out else ''}\n"
        f"Exit code: {returncode}\n"
        f"{'=' * 60}\n\n"
    )
    log_file.write_text(header + readable)

    return {"returncode": returncode, "elapsed": round(elapsed, 1), "timed_out": timed_out}


# ── Stream-json log parser ────────────────────────────────────────────────────


def parse_stream_to_log(raw_lines: list[str]) -> str:
    parts = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            parts.append(line)
            continue

        msg_type = msg.get("type", "")
        if msg_type == "assistant":
            for block in msg.get("message", {}).get("content", []):
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        parts.append(f"[Assistant]\n{block['text']}\n")
                    elif block.get("type") == "tool_use":
                        tool_name = block.get("name", "?")
                        tool_input = json.dumps(block.get("input", {}), indent=2, ensure_ascii=False)
                        parts.append(f"[Tool Call: {tool_name}]\n{tool_input}\n")
        elif msg_type == "tool":
            content = msg.get("content", "")
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(f"[Tool Result]\n{c['text'][:2000]}\n")
                    elif isinstance(c, str):
                        parts.append(f"[Tool Result]\n{c[:2000]}\n")
            elif isinstance(content, str):
                parts.append(f"[Tool Result]\n{content[:2000]}\n")
        elif msg_type == "result":
            cost = msg.get("cost_usd", "?")
            duration = msg.get("duration_ms", "?")
            parts.append(f"\n--- Result ---\ncost_usd: {cost}\nduration_ms: {duration}\n")
    return "\n".join(parts)


# ── Post-pipeline helpers ─────────────────────────────────────────────────────


def _execute_and_save_csv(sql: str, db_id: str, db_path: str, out_csv: Path) -> None:
    import snowflake.connector
    cred_path = os.path.join(db_path, "snowflake_credential.json")
    with open(cred_path) as f:
        creds = json.load(f)
    conn = snowflake.connector.connect(**creds, database=db_id)
    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        headers = [desc[0] for desc in cursor.description]
        rows = cursor.fetchall()
        with open(out_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(rows)
    finally:
        conn.close()


# ── Independent verifier ──────────────────────────────────────────────────────


def _csv_preview(csv_path: Path, max_rows: int = 10) -> tuple[str, int]:
    """Return (preview_text, total_row_count)."""
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        return "(empty — no CSV was produced)", 0
    with open(csv_path) as f:
        lines = f.read().splitlines()
    if not lines:
        return "(empty)", 0
    header = lines[0]
    data = lines[1:]
    shown = data[:max_rows]
    total = len(data)
    preview = header + "\n" + "\n".join(shown)
    if total > max_rows:
        preview += f"\n... ({total - max_rows} more rows not shown)"
    return preview, total


def run_verifier(
    question: dict,
    csv_path: Path,
    verifier_file: Path,
    claude_path: str = "claude",
    model: str = "sonnet",
    timeout: int = 180,
) -> dict:
    """Run an independent verification pass that sees ONLY the question and the
    CSV output — no SQL, no plan, no tool access, no hooks, no skills.

    Returns verdict dict with keys: verdict ("correct"/"incorrect"/"uncertain"), reason.
    Also writes the verdict to *verifier_file*.
    """
    instruction = question.get("instruction", "")
    db_id = question.get("db_id", "")
    preview, total_rows = _csv_preview(csv_path, max_rows=10)

    prompt = (
        "You are an independent verification agent for a text-to-SQL benchmark. "
        "Your ONLY job is to decide whether the CSV output below correctly answers "
        "the question below. You have NO access to the database, the SQL query, or "
        "the previous agent's reasoning — only the question text and the CSV output.\n\n"
        "Read the question carefully, then read the CSV. Decide:\n"
        "1. Does the output have the columns the question literally asks for, in "
        "the order and with the names it specifies? Extra or missing columns are "
        "mistakes.\n"
        "2. Is the row count plausible for what the question asks (single row vs "
        "per-group vs ranked top-N, etc.)?\n"
        "3. Do any of the values look implausible — absurd magnitudes, negative "
        "values where positive are required, dates outside a sensible range, or "
        "values that don't match what the question literally describes?\n"
        "4. Does every clause of the question appear to be satisfied by the "
        "output? If the question specifies a filter, grouping, or ordering and "
        "the output visibly violates it, flag that.\n\n"
        "Return a single-line JSON verdict. Do not print anything else — no "
        "preamble, no markdown fences, no commentary.\n\n"
        '{"verdict": "correct" | "incorrect" | "uncertain", "reason": "one sentence"}\n\n'
        f"----- QUESTION (db: {db_id}) -----\n{instruction}\n\n"
        f"----- CSV OUTPUT ({total_rows} total rows) -----\n{preview}\n\n"
        "----- END -----"
    )

    cmd = [
        claude_path, "-p", prompt,
        "--output-format", "text",
        "--model", model,
        "--permission-mode", "bypassPermissions",
        "--disable-slash-commands",
    ]

    verdict: dict
    try:
        with tempfile.TemporaryDirectory() as tmp:
            proc = subprocess.run(
                cmd, cwd=tmp, capture_output=True, text=True, timeout=timeout,
            )
        raw = (proc.stdout or "").strip()
    except subprocess.TimeoutExpired:
        verdict = {"verdict": "uncertain", "reason": "verifier timed out"}
        verifier_file.write_text(json.dumps(verdict, indent=2))
        return verdict
    except Exception as e:
        verdict = {"verdict": "uncertain", "reason": f"verifier error: {e}"}
        verifier_file.write_text(json.dumps(verdict, indent=2))
        return verdict

    # Parse the JSON verdict out of the raw output
    verdict = None
    try:
        verdict = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'\{[^{}]*"verdict"\s*:\s*"(?:correct|incorrect|uncertain)"[^{}]*\}', raw)
        if m:
            try:
                verdict = json.loads(m.group())
            except json.JSONDecodeError:
                pass

    if not isinstance(verdict, dict) or "verdict" not in verdict:
        verdict = {
            "verdict": "uncertain",
            "reason": "could not parse verifier output",
            "raw": raw[:500],
        }

    verifier_file.write_text(json.dumps(verdict, indent=2))
    return verdict


# ── Main pipeline ─────────────────────────────────────────────────────────────


def run_question(
    question: dict,
    output_dir: Path,
    db_path: str = "spider2-snow",
    timeout: int = 900,
    claude_path: str = "claude",
    model: str = "sonnet",
    effort: str | None = None,
    skip_existing: bool = True,
    mcp_url: str | None = None,
    **_ignored,  # absorb legacy kwargs (e.g. max_retries) from run_fold.py callers
) -> dict:
    """Run one Claude Code agent loop on a single question."""
    instance_id = question.get("instance_id") or question.get("question_id", "unknown")
    db_id = question.get("db_id", "")

    q_output_dir = output_dir / str(instance_id)
    q_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = q_output_dir / "logs"
    logs_dir.mkdir(exist_ok=True)

    answer_file = q_output_dir / "answer.sql"

    if skip_existing and answer_file.exists() and answer_file.stat().st_size > 0:
        return {"instance_id": instance_id, "status": "skipped", "message": "answer.sql already exists"}

    if not WORKFLOW_PATH.exists():
        return {
            "instance_id": instance_id, "status": "error",
            "message": f"workflow.md not found at {WORKFLOW_PATH}",
        }
    system_append = WORKFLOW_PATH.read_text(encoding="utf-8")

    env = {
        **os.environ,
        "CCSQL_DB_ID": db_id,
        "CCSQL_DB_PATH": db_path,
        "CCSQL_DB_TYPE": "snowflake",
        "CCSQL_QDIR": str(q_output_dir.resolve()),
    }

    prompt = build_prompt(question, db_path, answer_file)

    # MCP config: if a persistent HTTP server URL was provided, generate a per-
    # question .mcp.json that points at it with the right db_id/session_id
    # headers. Otherwise, stay silent and let claude spawn its own stdio server
    # from the repo-root .mcp.json (legacy path).
    mcp_cfg_path: Path | None = None
    mcp_cfg_tmpdir: tempfile.TemporaryDirectory | None = None
    if mcp_url:
        mcp_cfg_tmpdir = tempfile.TemporaryDirectory(prefix=f"ccsql-mcp-{instance_id}-")
        mcp_cfg_path = Path(mcp_cfg_tmpdir.name) / "mcp.json"
        write_mcp_config(
            mcp_cfg_path,
            url=mcp_url,
            db_id=db_id,
            session_id=f"{instance_id}-{uuid.uuid4().hex[:8]}",
        )
        # With HTTP transport the env var is unused by the server, but other
        # code (e.g. verifier fallbacks) may still read it, so keep setting it.

    t_start = time.time()
    print(f"    running agent ({timeout}s budget) ...", flush=True)
    try:
        run_result = _run_claude(
            prompt, system_append, env, logs_dir,
            timeout, claude_path, model, effort,
            mcp_config_path=mcp_cfg_path,
        )
    finally:
        if mcp_cfg_tmpdir is not None:
            mcp_cfg_tmpdir.cleanup()
    total_elapsed = round(time.time() - t_start, 1)

    pipeline = {
        "instance_id": instance_id,
        "db_id": db_id,
        "model": model,
        "effort": effort,
        "total_elapsed": total_elapsed,
        "claude_returncode": run_result["returncode"],
        "timed_out": run_result["timed_out"],
    }

    # ── Post-processing ──────────────────────────────────────────────────────
    if answer_file.exists() and answer_file.stat().st_size > 0:
        try:
            sys.path.insert(0, str(CCSQL_ROOT / "src"))
            from db_tools import post_format_generated_query
            raw_sql = answer_file.read_text().strip()
            formatted_sql = post_format_generated_query(
                raw_sql, db_path=db_path, db_type="snowflake", include_comment=False,
            )
            (q_output_dir / "answer_formatted.sql").write_text(formatted_sql)
        except Exception as fmt_err:
            print(f"  [WARNING] {instance_id}: formatting failed: {fmt_err}", file=sys.stderr)
            formatted_sql = None

        # Execute — try formatted first, fall back to raw
        csv_path = q_output_dir / f"{instance_id}.csv"
        raw_sql = answer_file.read_text().strip()
        formatted_path = q_output_dir / "answer_formatted.sql"
        formatted_sql = formatted_path.read_text().strip() if formatted_path.exists() else None

        exec_ok = False
        for label, sql in [("formatted", formatted_sql), ("raw", raw_sql)]:
            if sql is None:
                continue
            try:
                _execute_and_save_csv(sql, db_id, db_path, csv_path)
                exec_ok = True
                break
            except Exception as exec_err:
                print(f"  [WARNING] {instance_id}: {label} query execution failed: {exec_err}", file=sys.stderr)
                if label == "formatted":
                    print(f"  [INFO] {instance_id}: retrying with raw model-generated SQL", file=sys.stderr)

        # ── Independent verifier (logger-only — does not modify answer.sql) ──
        if exec_ok:
            verifier_file = q_output_dir / "verifier.json"
            try:
                verdict = run_verifier(
                    question=question,
                    csv_path=csv_path,
                    verifier_file=verifier_file,
                    claude_path=claude_path,
                    model=model,
                    timeout=180,
                )
                pipeline["verifier_verdict"] = verdict
            except Exception as ver_err:
                pipeline["verifier_error"] = str(ver_err)

        if run_result["timed_out"]:
            status = "success_partial"
            message = f"answer.sql written ({answer_file.stat().st_size}B) but agent hit wall-clock timeout in {total_elapsed:.0f}s"
        elif not exec_ok:
            status = "success_partial"
            message = f"answer.sql written ({answer_file.stat().st_size}B) but execution failed in {total_elapsed:.0f}s"
        else:
            status = "success"
            message = f"answer.sql written ({answer_file.stat().st_size}B) in {total_elapsed:.0f}s"
    else:
        if run_result["timed_out"]:
            status = "timeout"
            message = f"no answer.sql; agent hit wall-clock timeout in {total_elapsed:.0f}s"
        else:
            status = "no_output"
            message = f"no answer.sql produced in {total_elapsed:.0f}s (exit={run_result['returncode']})"

    pipeline["status"] = status
    pipeline["message"] = message
    with open(q_output_dir / "pipeline.json", "w") as f:
        json.dump(pipeline, f, indent=2, default=str)

    return {"instance_id": instance_id, "status": status, "message": message}


# ── CLI entry point ───────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Run a single-shot Claude Code agent on one text-to-SQL question."
    )
    parser.add_argument("--question_json", required=True,
                        help="Question as a JSON string or @path_to_json_file")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--db_path", default="spider2-snow")
    parser.add_argument("--timeout", type=int, default=900,
                        help="Wall-clock timeout per question in seconds (default: 900)")
    parser.add_argument("--claude_path", default="claude")
    parser.add_argument("--model", default="sonnet")
    parser.add_argument("--effort", default=None)
    parser.add_argument("--no_skip", action="store_true",
                        help="Re-run even if answer.sql exists")
    args = parser.parse_args()

    qj = args.question_json
    if qj.startswith("@"):
        with open(qj[1:]) as f:
            question = json.load(f)
    else:
        question = json.loads(qj)

    os.environ["CCSQL_DB_PATH"] = args.db_path

    # For standalone single-question runs, spin up our own short-lived HTTP MCP
    # server so the agent sees tools immediately (no stdio cold-start race).
    with running_mcp_server(db_path=args.db_path, db_type="snowflake") as h:
        result = run_question(
            question=question,
            output_dir=args.output_dir,
            db_path=args.db_path,
            timeout=args.timeout,
            claude_path=args.claude_path,
            model=args.model,
            effort=args.effort,
            skip_existing=not args.no_skip,
            mcp_url=h.url,
        )
    print(json.dumps(result))
    if result["status"] not in ("success", "skipped"):
        sys.exit(1)


if __name__ == "__main__":
    main()
