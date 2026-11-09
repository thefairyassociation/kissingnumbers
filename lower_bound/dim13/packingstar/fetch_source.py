#!/usr/bin/env python3
"""Fetch and verify the pinned PackingStar dimension-13 source archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
import tempfile
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


DEFAULT_LOCK = Path(__file__).with_name("source_lock.json")


def load_lock(path: Path) -> dict[str, str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    required = ("repository", "commit", "archive_path", "archive_sha256")
    missing = [key for key in required if not isinstance(payload.get(key), str)]
    if missing:
        raise ValueError(f"source lock is missing string fields: {', '.join(missing)}")
    commit = payload["commit"]
    digest = payload["archive_sha256"]
    if len(commit) != 40 or any(ch not in "0123456789abcdef" for ch in commit):
        raise ValueError(f"source commit is not a full lowercase SHA-1: {commit!r}")
    if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
        raise ValueError(f"archive hash is not a lowercase SHA-256: {digest!r}")
    return {key: payload[key] for key in required}


def archive_url(lock: dict[str, str]) -> str:
    repository = lock["repository"]
    if repository.count("/") != 1:
        raise ValueError(f"invalid GitHub repository name: {repository!r}")
    quoted_path = urllib.parse.quote(lock["archive_path"], safe="/")
    return (
        f"https://raw.githubusercontent.com/{repository}/{lock['commit']}/"
        f"{quoted_path}"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_locked_archive(lock: dict[str, str], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        archive_url(lock),
        headers={"User-Agent": "kissingnumbers-exact-verifier/1"},
    )
    with tempfile.NamedTemporaryFile(
        dir=destination.parent, prefix=destination.name + ".", delete=False
    ) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                shutil.copyfileobj(response, temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
    temporary_path.replace(destination)


def verify_archive(lock: dict[str, str], archive: Path) -> str:
    actual = sha256_file(archive)
    expected = lock["archive_sha256"]
    if actual != expected:
        raise ValueError(
            f"PackingStar archive SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return actual


def _is_symlink(info: zipfile.ZipInfo) -> bool:
    return stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)


def extract_safely(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            if _is_symlink(info):
                raise ValueError(f"refusing symlink in source archive: {info.filename!r}")
            target = (root / info.filename).resolve()
            if os.path.commonpath((str(root), str(target))) != str(root):
                raise ValueError(f"refusing path traversal in source archive: {info.filename!r}")
        zipped.extractall(root)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock-file", type=Path, default=DEFAULT_LOCK)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-dir", type=Path, required=True)
    parser.add_argument(
        "--offline",
        action="store_true",
        help="verify an existing archive instead of downloading it",
    )
    args = parser.parse_args(argv)

    try:
        lock = load_lock(args.lock_file)
        if not args.offline:
            download_locked_archive(lock, args.archive)
        if not args.archive.is_file():
            raise FileNotFoundError(args.archive)
        digest = verify_archive(lock, args.archive)
        if args.extract_dir.exists():
            shutil.rmtree(args.extract_dir)
        extract_safely(args.archive, args.extract_dir)
    except Exception as exc:
        print(f"source fetch failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "repository": lock["repository"],
                "commit": lock["commit"],
                "archive_path": lock["archive_path"],
                "archive_sha256": digest,
                "archive": str(args.archive),
                "extract_dir": str(args.extract_dir),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
