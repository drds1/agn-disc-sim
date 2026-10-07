"""
Push batches to Kaggle, wait for them, download their output, and keep a
ledger of the compute used (remote/usage.json, git-ignored).

Usage (from the repository root):
    poetry run python remote/runs.py push test e2_q5e-3   # build and push
    poetry run python remote/runs.py wait                 # poll every minute until all finish; download each
                                                          # into outputs/kaggle/<batch>_v<N>/
    poetry run python remote/runs.py usage                # hours used: per run, last 7 days, this month

Runtime is the session's own clock (the last timestamp in the kernel log), so
it counts setup as well as simulating.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "outputs" / "kaggle"
LEDGER = HERE / "usage.json"
POLL_SECONDS = 60
sys.path.insert(0, str(HERE))

import build  # noqa: E402


def kaggle(*args) -> str:
    return subprocess.run(["kaggle", *args], capture_output=True, text=True, check=True).stdout


def load() -> list:
    return json.loads(LEDGER.read_text()) if LEDGER.exists() else []


def save(runs: list) -> None:
    LEDGER.write_text(json.dumps(runs, indent=1))


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _push_one(run: dict) -> bool:
    """Build and push ``run``'s batch; False if Kaggle's concurrent-session
    limit (5 batch CPU sessions) is reached, so it stays queued."""
    out = build.build(run["batch"], build.kaggle_username(), HERE / "build")
    run["slug"] = json.loads((out / "kernel-metadata.json").read_text())["id"]
    text = kaggle("kernels", "push", "-p", str(out))
    if "Maximum batch CPU session" in text:
        return False
    match = re.search(r"version (\d+)", text)
    if match is None:
        raise RuntimeError(f"unexpected push output for {run['batch']}: {text}")
    run.update(version=int(match.group(1)), pushed=now(), status="pushed", code=build.git_state())
    print(f"pushed {run['slug']} v{run['version']}", flush=True)
    return True


def push(batches) -> None:
    runs = load()
    for batch in batches:
        if batch not in build.BATCHES:
            raise SystemExit(f"unknown batch {batch!r}; choose from {sorted(build.BATCHES)}")
        run = dict(batch=batch, slug=None, version=None, pushed=None, queued=now(), status="queued")
        runs.append(run)
        if not _push_one(run):
            print(f"{batch}: Kaggle's session limit reached; queued (runs.py wait pushes it when a slot frees)",
                  flush=True)
        save(runs)


def finish(run: dict, status: str) -> None:
    """Download a finished run's output and record its runtime."""
    dest = OUT / f"{run['batch']}_v{run['version']}"
    # the tarball and the logs only: fetching hundreds of snapshots and frames one by one is slow and can hang
    subprocess.run(["kaggle", "kernels", "output", run["slug"], "-p", str(dest), "-o",
                    "--file-pattern", r"(outputs\.tar\.gz|\.log)$"], capture_output=True, text=True, check=True,
                   timeout=3600)
    if (dest / "outputs.tar.gz").exists():
        subprocess.run(["tar", "xzf", "outputs.tar.gz"], cwd=dest, check=True)
    logs = list(dest.glob("*.log"))
    events = json.loads(logs[0].read_text()) if logs else []
    run.update(status=status, finished=now(), output=str(dest.relative_to(HERE.parent)),
               hours=round(events[-1]["time"] / 3600, 3) if events else None)
    failed = [line for f in (dest / "outputs" / "logs").rglob("*.log") for line in f.read_text().splitlines()
              if "Traceback" in line]
    print(f"{run['slug']} v{run['version']}: {status}, {run['hours']} h -> {dest}"
          + (f", {len(failed)} traceback(s) in its logs" if failed else ""), flush=True)


def wait() -> None:
    while True:
        runs = load()
        pending = [r for r in runs if r["status"] in ("pushed", "running", "queued")]
        if not pending:
            break
        for run in [r for r in pending if r["status"] == "queued"]:
            if not _push_one(run):
                break  # still full; try again next poll
        for run in [r for r in pending if r["status"] in ("pushed", "running")]:
            status = kaggle("kernels", "status", run["slug"])
            # Status refers to the latest version, which is this run's if it was the last pushed.
            if "COMPLETE" in status:
                finish(run, "complete")
            elif "ERROR" in status or "CANCEL" in status:
                finish(run, "error")
            else:
                run["status"] = "running"
        save(runs)
        if any(r["status"] in ("pushed", "running", "queued") for r in runs):
            time.sleep(POLL_SECONDS)
    usage()


def usage() -> None:
    runs = load()
    t = datetime.now(timezone.utc)
    week, month = t - timedelta(days=7), t.replace(day=1, hour=0, minute=0, second=0)
    for r in runs:
        print(f"{(r['pushed'] or r.get('queued', ''))[:16]}  {r['slug'] or r['batch']} v{r['version']}  "
              f"{r['status']:9s} {r.get('hours') or 0:6.2f} h")

    def total(since):
        return sum(r.get("hours") or 0 for r in runs if r["pushed"] and datetime.fromisoformat(r["pushed"]) >= since)

    print(f"CPU hours: last 7 days {total(week):.2f}, this month {total(month):.2f}, all {total(t.replace(year=1)):.2f}")


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["usage"]
    {"push": lambda: push(rest), "wait": wait, "usage": usage}[command]()
