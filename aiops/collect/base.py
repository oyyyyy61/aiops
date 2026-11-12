"""采集层公共类型：统一采样结构与采集器接口。"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from ..faultcode.models import Target, Vendor, VendorEvent

# 指标名统一为 ``对象.指标``，与 Prometheus 的 snake_case 一一对应（见 metrics.py）。
# 对象：gpu / node / job / svc / obs


@dataclass
class Sample:
    """一次采样：时间戳 + 对象 + 指标字典。"""

    ts: float
    kind: str                     # gpu / node / job / svc
    target: Target
    metrics: dict[str, float] = field(default_factory=dict)
    labels: dict[str, str] = field(default_factory=dict)

    def get(self, key: str, default: float | None = None) -> float | None:
        return self.metrics.get(key, default)

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "kind": self.kind,
            "target": self.target.as_dict(),
            "metrics": {k: round(v, 6) for k, v in self.metrics.items()},
            "labels": self.labels,
        }


@dataclass
class Snapshot:
    """一轮采集的完整结果：GPU/节点/作业指标 + 厂商事件。"""

    ts: float
    samples: list[Sample] = field(default_factory=list)
    events: list[VendorEvent] = field(default_factory=list)
    source_up: bool = True
    notes: list[str] = field(default_factory=list)

    def extend(self, other: "Snapshot") -> None:
        self.samples.extend(other.samples)
        self.events.extend(other.events)
        self.source_up = self.source_up and other.source_up
        self.notes.extend(other.notes)

    def by_kind(self, kind: str) -> list[Sample]:
        return [s for s in self.samples if s.kind == kind]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ts": round(self.ts, 3),
            "source_up": self.source_up,
            "samples": [s.as_dict() for s in self.samples],
            "events": [
                {
                    "vendor": e.vendor.value,
                    "code": e.code,
                    "text": e.text,
                    "component": e.component,
                }
                for e in self.events
            ],
            "notes": self.notes,
        }


class Collector(Protocol):
    """采集器协议：一次 collect 返回一个 Snapshot。"""

    name: str

    def collect(self) -> Snapshot: ...


def num(value: Any, default: float | None = None) -> float | None:
    """解析 ``38 C`` / ``10.40 W`` / ``8163 MiB`` / ``N/A`` 这类 nvidia-smi 字段。"""

    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text or text.upper() in {"N/A", "NA", "NONE", "UNKNOWN", "[NOT SUPPORTED]"}:
        return default
    # 取前导数值（允许科学计数、负号、小数）
    token = ""
    for ch in text:
        if ch.isdigit() or ch in "+-.eE":
            token += ch
        elif token:
            break
    try:
        return float(token)
    except ValueError:
        return default


def boolish(value: Any) -> float:
    """Yes/No/Enabled/Disabled -> 0/1。"""

    text = str(value).strip().lower()
    return 1.0 if text in {"yes", "enabled", "true", "on", "active", "1", "ok"} else 0.0


def now() -> float:
    return time.time()


def write_jsonl(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class StaticCollector:
    """把固定快照包成采集器，用于单元测试与回放。"""

    name = "static"

    def __init__(self, snapshots: Sequence[Snapshot] | Snapshot):
        self._snapshots = list(snapshots) if not isinstance(snapshots, Snapshot) else [snapshots]
        self._idx = 0

    def collect(self) -> Snapshot:
        if not self._snapshots:
            return Snapshot(ts=now(), source_up=False)
        snap = self._snapshots[min(self._idx, len(self._snapshots) - 1)]
        self._idx += 1
        return snap
