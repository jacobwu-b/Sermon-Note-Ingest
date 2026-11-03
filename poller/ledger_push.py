"""Hand a job's ledger writes to the one job that holds the push key (ADR-0027).

The deploy key bypasses main's ruleset, so it never sits in a job that runs third-party code.
A producer (the poll job, each transcribe shard) ``stage``s the allowlisted ``data/`` files it
changed, with the commit it started from, and uploads them as an artifact. The push job
``apply``s every artifact onto the live tip by a field-level three-way merge, and
``check-staged`` refuses any other path before the commit. poll.yml's push job then regenerates
``data/README.md``'s stats from the merged ledgers (``stats``, ADR-0030): the README is written
there, never carried by an artifact. The push job runs this module as
``python -E -S``: without site-packages, so a third-party import on this path fails instead of
running.

``apply`` treats an artifact as untrusted input: it is only ever parsed as JSON, and only
allowlisted paths are written.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from poller import config, stats, store

logger = logging.getLogger("poller")

# Names the commit a producer started from; every other file in an artifact is a ledger write.
BASE_FILE = "ledger-base"
ALERT_STATE = "data/alerts/state.json"
# Written by the push job itself from the merged ledgers, so never admitted from an artifact.
STATS_README = "data/README.md"
_LEDGER_RE = re.compile(rf"data/{config.CHURCH_KEY_RE.pattern.strip('^$')}\.json")
_SHA_RE = re.compile(r"[0-9a-f]{40}")
_MISSING = object()

Records = dict[str, dict[str, Any]]


class LedgerPushError(Exception):
    """A ledger write the push job must refuse rather than commit."""


def allowed(path: str) -> bool:
    """Whether automation may push ``path``: a church ledger or the discovery alert state."""
    return path == ALERT_STATE or _LEDGER_RE.fullmatch(path) is not None


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, check=False)


def _git_out(repo: Path, *args: str) -> str:
    result = _git(repo, *args)
    if result.returncode != 0:
        raise LedgerPushError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout


def stage(repo: Path, out: Path) -> list[str]:
    """Copy every ``data/`` file changed since ``repo``'s HEAD into ``out``, with HEAD's sha."""
    changed = _git_out(repo, "diff", "--name-only", "-z", "HEAD", "--", "data/").split("\0")
    changed += _git_out(repo, "ls-files", "--others", "--exclude-standard", "-z", "--", "data/").split("\0")
    paths = sorted({path for path in changed if path})
    refused = [path for path in paths if not allowed(path) or not (repo / path).is_file()]
    if refused:
        raise LedgerPushError(f"changed outside the ledger allowlist, or deleted: {', '.join(refused)}")
    out.mkdir(parents=True, exist_ok=True)
    (out / BASE_FILE).write_text(_git_out(repo, "rev-parse", "HEAD"))
    for path in paths:
        (out / path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / path, out / path)
    logger.info("staged %d ledger file(s) against %s", len(paths), (out / BASE_FILE).read_text().strip())
    return paths


def _parse(text: str, where: str) -> Records:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise LedgerPushError(f"{where}: not JSON: {exc}") from exc
    if not isinstance(data, dict) or not all(isinstance(v, dict) for v in data.values()):
        raise LedgerPushError(f"{where}: not an object of records keyed by guid or church")
    return data


def _read_artifact(root: Path) -> tuple[str, dict[str, Records]]:
    """One artifact's base sha and its files, every path and shape checked before anything is written."""
    base = (root / BASE_FILE).read_text().strip()
    if not _SHA_RE.fullmatch(base):
        raise LedgerPushError(f"{root}: base {base!r} is not a full commit sha")
    files: dict[str, Records] = {}
    for entry in sorted(root.rglob("*")):
        rel = entry.relative_to(root).as_posix()
        if entry.is_dir() and not entry.is_symlink():
            continue
        if rel == BASE_FILE:
            continue
        if entry.is_symlink() or not entry.is_file() or not allowed(rel):
            raise LedgerPushError(f"{root}: {rel} is outside the ledger allowlist")
        files[rel] = _parse(entry.read_text(encoding="utf-8"), f"{root.name}:{rel}")
    return base, files


def _merge_fields(base: dict[str, Any], ours: dict[str, Any], theirs: dict[str, Any], where: str) -> dict:
    merged = dict(theirs)
    for field in [*ours, *(f for f in base if f not in ours)]:
        b, o, t = base.get(field, _MISSING), ours.get(field, _MISSING), theirs.get(field, _MISSING)
        if o == b or o == t:
            continue
        if t != b:
            raise LedgerPushError(
                f"{where}: field {field!r} changed on both sides ({o!r} here, {t!r} on the tip)"
            )
        if o is _MISSING:
            del merged[field]
        else:
            merged[field] = o
    return merged


def merge(base: Records, ours: Records, theirs: Records, where: str) -> Records:
    """``theirs`` with every record and field ``ours`` changed from ``base`` applied onto it.

    A field both sides changed to different values is refused, never silently resolved: the
    writers own disjoint fields and records (ADR-0005), so it means something is wrong.
    """
    merged = dict(theirs)
    for key in [*ours, *(k for k in base if k not in ours)]:
        b, o, t = base.get(key), ours.get(key), theirs.get(key)
        if o == b or o == t:
            continue
        if t == b:
            if o is None:
                del merged[key]
            else:
                merged[key] = o
            continue
        if o is None or t is None:
            raise LedgerPushError(f"{where}: {key!r} removed on one side and changed on the other")
        merged[key] = _merge_fields(b or {}, o, t, f"{where} {key!r}")
    return merged


def _base_text(repo: Path, base: str, path: str) -> str | None:
    if _git(repo, "cat-file", "-e", f"{base}^{{commit}}").returncode != 0:
        # A shallow checkout of the tip lacks an older base; GitHub serves any reachable sha.
        _git_out(repo, "fetch", "--quiet", "--no-tags", "--depth=1", "origin", base)
    if _git(repo, "cat-file", "-e", f"{base}:{path}").returncode != 0:
        return None
    return _git_out(repo, "show", f"{base}:{path}")


def apply(repo: Path, root: Path) -> int:
    """Merge every artifact under ``root`` onto ``repo``'s working tree; return how many."""
    artifacts = [_read_artifact(marker.parent) for marker in sorted(root.rglob(BASE_FILE))]
    pending: dict[str, Records] = {}
    for base, files in artifacts:
        for path, ours in files.items():
            base_text = _base_text(repo, base, path)
            base_records = _parse(base_text, f"{base}:{path}") if base_text is not None else {}
            target = repo / path
            if path not in pending:
                pending[path] = _parse(target.read_text(encoding="utf-8"), path) if target.exists() else {}
            pending[path] = merge(base_records, ours, pending[path], path)

    # Every artifact merged cleanly before the first write, so a refusal leaves the tree untouched.
    for path, records in pending.items():
        ordered = (
            dict(sorted(records.items())) if path == ALERT_STATE else dict(store._ordered_items(records))
        )
        target = repo / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(ordered, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("merged %d artifact(s) into %d file(s)", len(artifacts), len(pending))
    return len(artifacts)


def check_staged(repo: Path) -> list[str]:
    """Every staged path outside the ledger allowlist, besides the stats README this job writes."""
    staged = _git_out(repo, "diff", "--cached", "--name-only", "-z").split("\0")
    return [path for path in staged if path and path != STATS_README and not allowed(path)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("stage", help="Copy this checkout's ledger changes into OUT.").add_argument("out")
    commands.add_parser("apply", help="Merge every artifact under ROOT onto this checkout.").add_argument(
        "root"
    )
    commands.add_parser("check-staged", help="Fail if anything outside the ledger allowlist is staged.")
    commands.add_parser("stats", help="Regenerate data/README.md's stats from this checkout's ledgers.")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=config.load_log_level(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    repo = Path.cwd()
    try:
        if args.command == "stage":
            stage(repo, Path(args.out))
        elif args.command == "apply":
            apply(repo, Path(args.root))
        elif args.command == "stats":
            stats.regenerate()
        else:
            refused = check_staged(repo)
            if refused:
                raise LedgerPushError(f"staged outside the ledger allowlist: {', '.join(refused)}")
    except (LedgerPushError, stats.StatsError) as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
