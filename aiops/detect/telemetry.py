"""观测质量门控：没有证据时，宁可不动作。

这条规则来自一个现实教训：exporter 掉线时，控制面看到的"一切正常"和"全挂了"
长得一模一样。把 telemetry 状态显式建模后，控制决策可以随证据强度降级：

===========  ==============================================
状态          允许的动作
===========  ==============================================
OK           全部
DEGRADED     只读 / 可逆动作（保存检查点、摘流、重试单步），不做迁移与隔离
STALE        仅告警 + 紧急保存，禁止任何拓扑变更
BLIND        仅告警与升级，动作一律拒绝（fail closed）
===========  ==============================================
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Mapping

from ..collect.base import Snapshot

# 可按证据强度放行的动作分组
REVERSIBLE_ACTIONS = {
    "notify_only",
    "escalate_human",
    "emergency_checkpoint",
    "drain_instance",
    "retry_step",
    "reduce_load",
    "reduce_batch",
}
SAFE_ANY_STATE = {"notify_only", "escalate_human", "emergency_checkpoint"}

STATE_ORDER = {"OK": 0, "DEGRADED": 1, "STALE": 2, "BLIND": 3}


@dataclass
class QualityReport:
    state: str = "OK"
    score: float = 1.0
    reasons: list[str] = field(default_factory=list)
    stale_sources: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.state == "OK"

    @property
    def disruptive_allowed(self) -> bool:
        return self.state == "OK"

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "score": round(self.score, 3),
            "reasons": self.reasons,
            "stale_sources": self.stale_sources,
        }


class TelemetryMonitor:
    """按数据源跟踪新鲜度，输出整体质量。

    ``expected_interval_s`` 是采集周期；超过 ``stale_factor`` 倍没更新记 STALE，
    超过 ``blind_factor`` 倍记 BLIND。多个源里最差的那个决定整体状态。
    """

    def __init__(
        self,
        sources: list[str] | None = None,
        expected_interval_s: float = 15.0,
        stale_factor: float = 3.0,
        blind_factor: float = 8.0,
    ):
        self.expected_interval_s = expected_interval_s
        self.stale_s = expected_interval_s * stale_factor
        self.blind_s = expected_interval_s * blind_factor
        self._last_seen: dict[str, float] = {s: 0.0 for s in (sources or [])}
        self._default_source = (sources or ["primary"])[0]
        self._missing_ratio: dict[str, float] = {}
        self._conflicts: list[str] = []
        self._started = time.time()

    def register(self, source: str) -> None:
        self._last_seen.setdefault(source, 0.0)

    def observe(self, snap: Snapshot, source: str | None = None) -> None:
        source = source or self._default_source
        self.register(source)
        self._last_seen[source] = snap.ts or time.time()
        expected = max(1, len(snap.samples))
        missing = sum(1 for sample in snap.samples if not sample.metrics)
        self._missing_ratio[source] = missing / expected
        self._conflicts = list(snap.notes)

    def record_missing(self, source: str, ratio: float) -> None:
        self.register(source)
        self._missing_ratio[source] = ratio

    def evaluate(self, now: float | None = None) -> QualityReport:
        now = now if now is not None else time.time()
        if not self._last_seen:
            return QualityReport(state="BLIND", score=0.0, reasons=["没有任何数据源"])
        worst = "OK"
        reasons: list[str] = []
        stale_sources: list[str] = []
        for source, last in self._last_seen.items():
            age = now - last if last else now - self._started
            if age > self.blind_s:
                state = "BLIND"
                reasons.append(f"{source}: {age:.0f}s 未更新（盲区）")
                stale_sources.append(source)
            elif age > self.stale_s:
                state = "STALE"
                reasons.append(f"{source}: {age:.0f}s 未更新（陈旧）")
                stale_sources.append(source)
            else:
                state = "OK"
            if STATE_ORDER[state] > STATE_ORDER[worst]:
                worst = state
            missing = self._missing_ratio.get(source, 0.0)
            if missing > 0.2:
                reasons.append(f"{source}: 指标缺失率 {missing:.0%}")
                if STATE_ORDER[worst] < STATE_ORDER["DEGRADED"]:
                    worst = "DEGRADED"
        if self._conflicts:
            reasons.extend(f"指标冲突: {c}" for c in self._conflicts)
            if STATE_ORDER[worst] < STATE_ORDER["DEGRADED"]:
                worst = "DEGRADED"
        score = {"OK": 1.0, "DEGRADED": 0.7, "STALE": 0.3, "BLIND": 0.0}[worst]
        return QualityReport(state=worst, score=score, reasons=reasons, stale_sources=stale_sources)


class ActionGate:
    """把观测质量映射成"这个动作现在准不准做"。"""

    def __init__(self, monitor: TelemetryMonitor):
        self.monitor = monitor

    def allowed(self, action: str, quality: QualityReport | None = None) -> tuple[bool, str]:
        q = quality or self.monitor.evaluate()
        if q.state == "OK":
            return True, "telemetry OK"
        if action in SAFE_ANY_STATE:
            return True, f"只读动作，允许（{q.state}）"
        if q.state == "DEGRADED" and action in REVERSIBLE_ACTIONS:
            return True, f"可逆动作，允许（{q.state}）"
        return False, f"观测状态 {q.state}，拒绝破坏性动作 {action}"
