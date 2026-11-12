"""检测层：指标阈值规则 + 厂商事件翻译，统一成带生命周期的 Fault。

两个刻意的设计：

1. **去抖与迟滞**：``for_samples`` 连续成立才算故障，``clear_samples`` 连续不成立才
   算恢复。否则一次采样抖动就会触发重启，把"检测"变成"抖动放大器"。
2. **事件优先**：同一对象同一时刻既命中精确厂商码、又命中指标阈值时，保留厂商码
   （信息量更大），指标只作为证据补充。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from ..collect.base import Sample, Snapshot
from ..faultcode.models import Fault, FaultState, Severity, Target, Vendor, VendorEvent
from ..faultcode.normalizer import Normalizer
from ..faultcode.taxonomy import Catalog


@dataclass
class _RuleState:
    breach: int = 0
    clear: int = 0
    onset: float = 0.0
    last_value: float | None = None


@dataclass
class Detection:
    new: list[Fault] = field(default_factory=list)
    updated: list[Fault] = field(default_factory=list)
    resolved: list[Fault] = field(default_factory=list)
    active: list[Fault] = field(default_factory=list)

    def all_codes(self) -> list[str]:
        return sorted({f.code.id for f in self.active})


class Detector:
    """有状态的故障检测器。同一实例应按时间顺序喂 Snapshot。"""

    def __init__(
        self,
        catalog: Catalog,
        normalizer: Normalizer | None = None,
        event_ttl_s: float = 900.0,
    ):
        self.catalog = catalog
        self.normalizer = normalizer or Normalizer(catalog)
        self.event_ttl_s = event_ttl_s
        self.active: dict[str, Fault] = {}
        self._rule_state: dict[str, _RuleState] = {}
        self._rules: list[tuple[object, object]] = [
            (code, rule) for code in catalog for rule in code.signals
        ]
        self._event_only: dict[str, Fault] = {}

    # ---------- 主入口 ----------

    def observe(self, snap: Snapshot, now: float | None = None) -> Detection:
        now = now if now is not None else time.time()
        det = Detection()

        for sample in snap.samples:
            self._observe_sample(sample, now, det)

        for event in snap.events:
            self._observe_event(event, now, det)

        self._expire_events(now, det)
        det.active = sorted(self.active.values(), key=lambda f: (f.severity.rank, f.code.id))
        return det

    # ---------- 指标规则 ----------

    def _observe_sample(self, sample: Sample, now: float, det: Detection) -> None:
        if not sample.metrics:
            return
        for code, rule in self._rules:
            value = sample.metrics.get(rule.metric)
            if value is None:
                continue
            key = f"{code.id}|{sample.target.key()}|{rule.metric}"
            state = self._rule_state.setdefault(key, _RuleState())
            prev = state.last_value
            state.last_value = value
            breached = rule.evaluate(value, prev)
            if breached:
                state.breach += 1
                state.clear = 0
                if state.breach == 1:
                    state.onset = sample.ts or now
            else:
                state.clear += 1
                state.breach = 0

            fp = f"{code.id}|{sample.target.key()}"
            fault = self.active.get(fp)
            if state.breach >= rule.for_samples and fault is None:
                fault = Fault(
                    code=code,
                    target=sample.target,
                    detected_at=now,
                    onset_at=state.onset or now,
                    detection_latency_s=max(0.0, now - (state.onset or now)),
                    evidence=[
                        f"{rule.metric}={value:g} {rule.op} {rule.value:g} "
                        f"(连续 {state.breach} 次)"
                    ],
                )
                self.active[fp] = fault
                det.new.append(fault)
                det.updated.append(fault)
            elif fault is not None and state.breach >= rule.for_samples:
                fault.evidence.append(f"{rule.metric}={value:g}")
                if fault not in det.updated:
                    det.updated.append(fault)
            elif fault is not None and state.clear >= rule.clear_samples:
                fault.resolve(now)
                self.active.pop(fp, None)
                det.resolved.append(fault)

    # ---------- 厂商事件 ----------

    def _observe_event(self, event: VendorEvent, now: float, det: Detection) -> None:
        for translation in self.normalizer.normalize_event(event, Target(vendor=event.vendor)):
            code = translation.fault.code
            # 事件本身不带完整对象时，用 component 作为 node 线索
            target = Target(node=event.component, vendor=event.vendor) if event.component else Target(
                vendor=event.vendor
            )
            fp = f"{code.id}|{target.key()}"
            existing = self.active.get(fp) or self._event_only.get(fp)
            if existing is not None:
                existing.evidence.extend(translation.fault.evidence)
                existing.evidence = existing.evidence[-8:]
                self._touch(existing, now)
                if existing not in det.updated:
                    det.updated.append(existing)
                continue
            new_fault = Fault(
                code=code,
                target=target,
                detected_at=now,
                onset_at=event.ts or now,
                detection_latency_s=max(0.0, now - (event.ts or now)),
                evidence=list(translation.fault.evidence),
            )
            self.active[fp] = new_fault
            self._event_only[fp] = new_fault
            self._touch(new_fault, now)
            det.new.append(new_fault)
            det.updated.append(new_fault)

    def _touch(self, fault: Fault, now: float) -> None:
        # 用 evidence 追加时间的技巧不好维护，单独记录最近一次事件时间
        setattr(fault, "_last_event_ts", now)

    def _expire_events(self, now: float, det: Detection) -> None:
        for fp, fault in list(self._event_only.items()):
            last = getattr(fault, "_last_event_ts", fault.onset_at)
            if now - last > self.event_ttl_s and fault.state != FaultState.RESOLVED:
                fault.resolve(now)
                self.active.pop(fp, None)
                self._event_only.pop(fp, None)
                det.resolved.append(fault)

    # ---------- 查询与运维辅助 ----------

    def active_by_severity(self, severity: Severity) -> list[Fault]:
        return [f for f in self.active.values() if f.severity == severity]

    def acknowledge(self, fingerprint: str) -> bool:
        fault = self.active.get(fingerprint)
        if fault is None:
            return False
        fault.acknowledge()
        return True

    def resolve(self, fingerprint: str, now: float | None = None) -> bool:
        fault = self.active.pop(fingerprint, None)
        self._event_only.pop(fingerprint, None)
        if fault is None:
            return False
        fault.resolve(now)
        return True

    def inject(self, fault: Fault) -> None:
        """测试/演练入口：直接注入一条故障。"""

        self.active[fault.fingerprint] = fault


class EventOnlyDetector:
    """只做厂商事件翻译的轻量检测器，用于"日志诊断"场景。"""

    def __init__(self, catalog: Catalog):
        self.normalizer = Normalizer(catalog)

    def diagnose(self, vendor: Vendor, lines: Iterable[str], node: str) -> list[Fault]:
        target = Target(node=node, vendor=vendor)
        return [t.fault for t in self.normalizer.normalize_lines(vendor, lines, target)]
