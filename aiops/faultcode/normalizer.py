"""厂商故障「翻译层」：把异构原始信号归一化成 UFC。

三层输入，同一个出口：

* 结构化事件（DCGM 的 Xid 字段、rocm-smi 的 RAS 字段、npu-smi 的 Error Code）
* 半结构化文本（dmesg / slog / 容器日志里的一行）
* 指标阈值（由 detect 层使用目录里的 signals 规则）

输出统一为 :class:`~aiops.faultcode.models.Fault`，带证据与置信度。
置信度不参与"是否告警"，但参与策略强度：assumed 映射只允许只读动作。
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Iterable

from .models import (
    Confidence,
    Fault,
    FaultCode,
    Target,
    Vendor,
    VendorEvent,
)
from .taxonomy import Catalog, Match

# dmesg 里常见的 Xid 行：NVRM: Xid (PCI:0000:01:00): 79, GPU has fallen off the bus.
_XID_RE = re.compile(r"Xid\s*\([^)]*\)\s*:\s*(\d+)", re.IGNORECASE)
_NVRM_XID_RE = re.compile(r"Xid.*?:\s*(\d+)", re.IGNORECASE)
_AMD_RING_RE = re.compile(r"amdgpu.*ring.*timeout", re.IGNORECASE)
_ASCEND_ERR_RE = re.compile(r"(?:Error\s*Code|err(?:or)?_?code)\s*[:=]\s*([0-9A-Fa-fx]+)")
_ASCEND_HEALTH_RE = re.compile(r"Health\s*Status\s*[:=]\s*(\w+)", re.IGNORECASE)


@dataclass
class Translation:
    """一次翻译结果，便于 CLI / 工单展示。"""

    vendor: Vendor
    raw: str
    fault: Fault
    mapping_confidence: Confidence
    via: str

    def as_dict(self) -> dict:
        return {
            "vendor": self.vendor.value,
            "raw": self.raw,
            "ufc": self.fault.code.id,
            "name": self.fault.code.name,
            "severity": self.fault.code.severity.value,
            "scope": self.fault.code.scope.value,
            "confidence": self.mapping_confidence.value,
            "via": self.via,
            "actions": list(self.fault.code.actions),
            "verify": list(self.fault.code.verify),
        }


class Normalizer:
    """把厂商事件或日志行翻译成统一故障。"""

    def __init__(self, catalog: Catalog):
        self.catalog = catalog
        self.unmapped: list[VendorEvent] = []

    # ---------- 单事件 ----------

    def normalize_event(self, event: VendorEvent, target: Target) -> list[Translation]:
        out: list[Translation] = []
        for match in self.catalog.match(event.vendor, event.code, event.text):
            out.append(self._translation(event.vendor, event.code or event.text, match, target, event))
        if not out:
            self.unmapped.append(event)
        return out

    def _translation(
        self, vendor: Vendor, raw: str, match: Match, target: Target, event: VendorEvent
    ) -> Translation:
        now = event.ts or time.time()
        fault = Fault(
            code=match.code,
            target=target,
            detected_at=now,
            onset_at=now,
            evidence=[f"{vendor.value}:{match.via}:{match.matched}"],
            detection_latency_s=0.0,
        )
        if event.text:
            fault.evidence.append(event.text.strip()[:240])
        return Translation(vendor, raw, fault, match.mapping.confidence, match.via)

    # ---------- 日志行 ----------

    def parse_log_line(self, vendor: Vendor, line: str, target: Target) -> list[Translation]:
        """从一行内核/设备日志里抽取厂商码并翻译。

        这是最实用的入口：现场往往只有 ``dmesg | grep -i xid`` 的输出。
        """

        events: list[VendorEvent] = []
        if vendor == Vendor.NVIDIA:
            m = _XID_RE.search(line) or _NVRM_XID_RE.search(line)
            if m:
                events.append(VendorEvent(vendor=vendor, code=m.group(1), text=line))
        elif vendor == Vendor.AMD:
            if _AMD_RING_RE.search(line):
                events.append(VendorEvent(vendor=vendor, code="RING_TIMEOUT", text=line))
            else:
                events.append(VendorEvent(vendor=vendor, text=line))
        elif vendor == Vendor.ASCEND:
            m = _ASCEND_ERR_RE.search(line)
            health = _ASCEND_HEALTH_RE.search(line)
            if health and health.group(1).upper() not in ("OK", "NORMAL"):
                events.append(
                    VendorEvent(vendor=vendor, code=f"HEALTH_{health.group(1).upper()}", text=line)
                )
            elif m:
                events.append(VendorEvent(vendor=vendor, code=m.group(1), text=line))
            else:
                events.append(VendorEvent(vendor=vendor, text=line))
        else:
            events.append(VendorEvent(vendor=vendor, text=line))

        out: list[Translation] = []
        for event in events:
            out.extend(self.normalize_event(event, target))
        return out

    # ---------- 批量 ----------

    def normalize_lines(
        self, vendor: Vendor, lines: Iterable[str], target: Target
    ) -> list[Translation]:
        out: list[Translation] = []
        for line in lines:
            out.extend(self.parse_log_line(vendor, line, target))
        return out

    def translate_raw(self, vendor: Vendor, raw: str, target: Target | None = None) -> list[Translation]:
        """按原始码查表（``Xid 79``、``HBM_ECC_UNCORRECTED``、``RAS_UE``）。"""

        target = target or Target(vendor=vendor)
        event = VendorEvent(vendor=vendor, code=raw, text=raw)
        return self.normalize_event(event, target)

def infer_target(code: FaultCode, node: str = "", gpu: str = "", job: str = "", service: str = "") -> Target:
    """按故障码 scope 补齐作用对象，避免 target 缺维度导致去重失效。"""

    return Target(
        node=node,
        gpu=gpu if code.scope.value in ("gpu",) else "",
        job=job,
        service=service if code.scope.value == "service" else "",
    )
