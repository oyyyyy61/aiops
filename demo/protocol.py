"""仿真作业与监督器之间的文件协议（心跳/尝试目录/控制文件）。

单独抽出来是因为两边都要用同一套定位规则：如果"第几次尝试"靠目录 mtime 猜，
控制面就可能读到上一次尝试的心跳，把"已结束"误判成"已完成"。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

POINTER = "current"


def hb_root(run_dir: str | Path) -> Path:
    root = Path(run_dir) / "hb"
    root.mkdir(parents=True, exist_ok=True)
    return root


def set_current_attempt(run_dir: str | Path, name: str) -> None:
    root = hb_root(run_dir)
    tmp = root / (POINTER + ".tmp")
    tmp.write_text(name, encoding="utf-8")
    for _ in range(40):
        try:
            os.replace(tmp, root / POINTER)
            return
        except PermissionError:
            import time

            time.sleep(0.002)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


def next_attempt_name(run_dir: str | Path) -> str:
    root = hb_root(run_dir)
    used: list[int] = []
    for path in root.glob("attempt-*"):
        try:
            used.append(int(path.name.split("-", 1)[1]))
        except (IndexError, ValueError):
            continue
    return f"attempt-{(max(used) + 1) if used else 1}"


def current_attempt_dir(run_dir: str | Path, create: bool = False) -> Path:
    root = hb_root(run_dir)
    pointer = root / POINTER
    if pointer.exists():
        try:
            name = pointer.read_text(encoding="utf-8").strip()
            path = root / name
            if path.is_dir():
                return path
        except OSError:
            pass
    dirs: list[tuple[int, Path]] = []
    for path in root.glob("attempt-*"):
        if not path.is_dir():
            continue
        try:
            dirs.append((int(path.name.split("-", 1)[1]), path))
        except (IndexError, ValueError):
            continue
    if dirs:
        return max(dirs, key=lambda item: item[0])[1]
    path = root / "attempt-1"
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
