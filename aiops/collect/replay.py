"""离线回放采集器：把 JSONL 遥测当作实时数据源。

没有第二、第三种硬件时，回放是唯一能"端到端跑通"的办法，也是回归测试的基线：
同一份故障剧本必须每次得到同样的决策，否则策略就是不可测的。
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

from ..faultcode.models import Target, Vendor, VendorEvent
from .base import Sample, Snapshot


@dataclass
class ReplayFrame:
    ts: float
    samples: list[Sample]
    events: list[VendorEvent]
    note: str = ""

    def to_snapshot(self) -> Snapshot:
        return Snapshot(ts=self.ts, samples=list(self.samples), events=list(self.events))


def _parse_sample(row: Mapping[str, Any], default_node: str = "") -> Sample:
    vendor = Vendor.parse(row.get("vendor", "generic"))
    target = Target(
        node=str(row.get("node", default_node)),
        gpu=str(row.get("gpu", "")),
        job=str(row.get("job", "")),
        service=str(row.get("service", "")),
        vendor=vendor,
    )
    metrics = {str(k): float(v) for k, v in (row.get("metrics") or {}).items()}
    labels = {str(k): str(v) for k, v in (row.get("labels") or {}).items()}
    labels.setdefault("vendor", vendor.value)
    return Sample(
        ts=float(row.get("ts", 0.0)),
        kind=str(row.get("kind", "gpu")),
        target=target,
        metrics=metrics,
        labels=labels,
    )


class ReplayCollector:
    """按帧回放 JSONL。

    文件格式（每行一帧）::

        {"ts": 0.0, "note": "steady", "samples": [...], "events": [...]}
    """

    name = "replay"

    def __init__(self, path: str | Path, node: str = "", loop: bool = False):
        self.path = Path(path)
        self.node = node
        self.loop = loop
        self.frames = self._load(self.path)
        self.cursor = 0

    def _load(self, path: Path) -> list[ReplayFrame]:
        frames: list[ReplayFrame] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            row = json.loads(line)
            frames.append(
                ReplayFrame(
                    ts=float(row.get("ts", len(frames))),
                    samples=[_parse_sample(s, self.node) for s in row.get("samples", [])],
                    events=[
                        VendorEvent(
                            vendor=Vendor.parse(e.get("vendor", row.get("vendor", "generic"))),
                            code=str(e.get("code", "")),
                            text=str(e.get("text", "")),
                            component=str(e.get("component", self.node)),
                        )
                        for e in row.get("events", [])
                    ],
                    note=str(row.get("note", "")),
                )
            )
        return frames

    def reset(self) -> None:
        self.cursor = 0

    def collect(self) -> Snapshot:
        if not self.frames:
            return Snapshot(ts=time.time(), source_up=False, notes=["回放文件为空"])
        if self.cursor >= len(self.frames):
            if not self.loop:
                return Snapshot(ts=self.frames[-1].ts, source_up=False, notes=["回放结束"])
            self.cursor = 0
        frame = self.frames[self.cursor]
        self.cursor += 1
        snap = frame.to_snapshot()
        snap.ts = time.time()  # 对外表现为"现在"，便于与实时链路共用检测器
        if frame.note:
            snap.notes.append(frame.note)
        return snap

    def __len__(self) -> int:
        return len(self.frames)


def write_replay(path: str | Path, frames: Iterable[Snapshot]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for snap in frames:
            fh.write(json.dumps(snap.as_dict(), ensure_ascii=False) + "\n")


def stream(frames: Iterable[Snapshot]) -> Iterator[Snapshot]:
    yield from frames
