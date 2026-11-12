"""恢复策略决策引擎。

输入：一条故障 + 作业上下文（有没有检查点、有没有备机、重试了几次、观测是否可信）。
输出：一个可审计的 Plan。

决策顺序（先安全、后强度）：

1. 观测不可信 -> 最多只能做只读/可逆动作；
2. 无自动动作（S4）-> 只观测；
3. 推理服务 -> 摘流重启；崩溃循环 -> 隔离升级；
4. 节点级致命（掉卡、PSU、宕机）-> 疏散；无备机则失败升级；
5. 慢节点 -> 按根因分流：硬件慢迁移/缩容，负载慢降载；
6. 作业级可恢复（挂起、NCCL 超时）-> 原地重启，重试耗尽后迁移；
7. 卡级可疑 -> 有备机迁移，否则原地重启，再不行缩容；
8. 容量类 -> 降载；
9. 其余 S3/S4 -> 只观测。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from ..detect.straggler import StragglerReport
from ..detect.telemetry import QualityReport
from ..faultcode.models import Fault, Scope, Severity, Target
from ..faultcode.taxonomy import Catalog
from . import strategies as S
from .strategies import Plan

# 被认为"节点级致命"的故障码：不允许原地重试
NODE_FATAL = {"UFC-GPU-001", "UFC-PWR-002", "UFC-NOD-001", "UFC-NOD-004", "UFC-DRV-001", "UFC-MEM-001"}
# 卡级硬件可疑：优先迁移
GPU_SUSPECT = {
    "UFC-GPU-002",
    "UFC-GPU-003",
    "UFC-GPU-005",
    "UFC-GPU-006",
    "UFC-GPU-007",
    "UFC-GPU-008",
    "UFC-MEM-003",
    "UFC-LNK-001",
    "UFC-LNK-002",
    "UFC-LNK-003",
    "UFC-LNK-005",
}
# 作业级可恢复（软件或等待类），可以先原地重启
JOB_RECOVERABLE = {"UFC-GPU-004", "UFC-JOB-001", "UFC-JOB-004", "UFC-JOB-006", "UFC-NET-001"}
# 容量类，降载即可
CAPACITY = {"UFC-MEM-004", "UFC-MEM-006", "UFC-JOB-003", "UFC-INF-002", "UFC-INF-003", "UFC-PWR-001", "UFC-THM-001", "UFC-THM-002"}


@dataclass
class JobContext:
    """决策所需的作业侧事实。缺哪个就按"最保守"假设处理。"""

    job: str
    kind: str = "train"              # train / inference
    node: str = ""
    world_size: int = 1
    checkpoint_available: bool = True
    checkpoint_age_s: float | None = 60.0
    retries_used: int = 0
    spare_nodes: int = 0
    telemetry: QualityReport | None = None
    straggler: StragglerReport | None = None
    straggler_cause: str = "unknown"  # hardware / load / unknown
    batch_scale: float = 1.0

    def bumped(self) -> "JobContext":
        return replace(self, retries_used=self.retries_used + 1)


class PolicyEngine:
    def __init__(self, catalog: Catalog, max_retries_cap: int = 3):
        self.catalog = catalog
        self.max_retries_cap = max_retries_cap

    # ---------- 单故障决策 ----------

    def decide(self, fault: Fault, ctx: JobContext) -> Plan:
        code = fault.code
        base = dict(
            fault_code=code.id,
            fault_name=code.name,
            severity=code.severity.value,
            target=fault.target.key(),
            actions=[],
        )

        # 1) 观测不可信：只允许安全动作
        if ctx.telemetry is not None and not ctx.telemetry.ok:
            plan = self._build(
                S.STRATEGY_OBSERVE,
                reason=f"观测质量 {ctx.telemetry.state}，拒绝破坏性动作；"
                f"原因：{'; '.join(ctx.telemetry.reasons[:2]) or '未知'}",
                **base,
            )
            plan.degraded_by_telemetry = True
            return plan

        # 2) 无自动动作
        if code.severity == Severity.S4 or code.scope == Scope.OBSERVABILITY:
            return self._build(S.STRATEGY_OBSERVE, reason="S4/观测类故障，仅记录", **base)

        # 3) 推理服务
        if ctx.kind == "inference" or code.scope == Scope.SERVICE:
            if code.id == "UFC-INF-004":
                return self._build(
                    S.STRATEGY_FAIL_FAST,
                    reason="推理实例崩溃循环，自动重启无法收敛，隔离 GPU 并升级",
                    **base,
                )
            if code.id == "UFC-INF-002" or code.id == "UFC-INF-003":
                return self._build(S.STRATEGY_DEGRADE, reason="推理过载/缓存劣化，降载观察", **base)
            return self._build(S.STRATEGY_DRAIN, reason="推理实例故障，摘流重启", **base)

        # 4) 节点级致命
        if code.id in NODE_FATAL:
            if ctx.spare_nodes > 0 and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_EVACUATE,
                    reason=f"{code.id} 为节点级致命故障，整机清空并报修",
                    **base,
                )
            return self._build(
                S.STRATEGY_FAIL_FAST,
                reason="节点级致命故障，但没有备机或可用检查点，停止自动动作并升级",
                **base,
            )

        # 5) 慢节点：按根因分流
        if code.id == "UFC-JOB-007":
            return self._decide_straggler(fault, ctx, **base)

        # 6) 作业级可恢复
        if code.id in JOB_RECOVERABLE:
            if self._retries_left(code, ctx) and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_RESTART,
                    reason=f"可恢复作业级故障，检查点年龄 {self._age_text(ctx)}，原地重启",
                    **base,
                )
            if ctx.spare_nodes > 0 and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_MIGRATE,
                    reason="重试预算耗尽，迁移到备用节点",
                    **base,
                )
            if ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_SHRINK,
                    reason="无备机但有检查点，缩容去坏 rank 续跑",
                    **base,
                )
            return self._build(
                S.STRATEGY_FAIL_FAST,
                reason="无检查点可恢复，失败升级（避免丢失全部进度）",
                **base,
            )

        # 7) 卡级可疑
        if code.id in GPU_SUSPECT:
            if ctx.spare_nodes > 0 and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_MIGRATE,
                    reason=f"{code.id} 指向单卡/链路可疑，迁移并隔离该卡",
                    **base,
                )
            if self._retries_left(code, ctx) and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_RESTART,
                    reason="无备机，先原地重启一次；连续复发再隔离",
                    **base,
                )
            if ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_SHRINK,
                    reason="反复出现且无备机，缩容跳过该卡",
                    **base,
                )
            return self._build(
                S.STRATEGY_FAIL_FAST,
                reason="卡级故障且无检查点/备机，升级人工",
                **base,
            )

        # 8) 容量类
        if code.id in CAPACITY:
            return self._build(S.STRATEGY_DEGRADE, reason="容量/散热类问题，降载继续", **base)

        # 9) 兜底
        if code.severity == Severity.S3:
            return self._build(
                S.STRATEGY_OBSERVE, reason="S3 故障，进入计划性处理窗口", **base
            )
        return self._build(
            S.STRATEGY_OBSERVE, reason="未匹配到专用策略，保守观测并升级", **base
        )

    # ---------- 批量决策 ----------

    def decide_all(self, faults: Iterable[Fault], ctx: JobContext) -> list[Plan]:
        plans = [self.decide(f, ctx) for f in faults]
        # 严重度高、动作强的排前面；同一作业只执行最强的一条破坏性动作
        plans.sort(key=lambda p: (Severity(p.severity).rank, 0 if p.is_noop else 1))
        return plans

    # ---------- 内部 ----------

    def _decide_straggler(self, fault: Fault, ctx: JobContext, **base) -> Plan:
        report = ctx.straggler
        cause = ctx.straggler_cause
        if cause == "hardware":
            if ctx.spare_nodes > 0 and ctx.checkpoint_available:
                return self._build(
                    S.STRATEGY_MIGRATE,
                    reason=f"微基准确认硬件慢（{self._slow_text(report)}），迁移该 rank 所在节点",
                    **base,
                )
            return self._build(
                S.STRATEGY_SHRINK,
                reason=f"硬件慢但无备机（{self._slow_text(report)}），缩容去掉慢 rank",
                **base,
            )
        if cause == "load":
            return self._build(
                S.STRATEGY_DEGRADE,
                reason=f"微基准正常，判定为负载不均（{self._slow_text(report)}），调数据不要换卡",
                **base,
            )
        if ctx.spare_nodes > 0 and ctx.checkpoint_available and self._retries_left(fault.code, ctx):
            return self._build(
                S.STRATEGY_MIGRATE,
                reason=f"慢节点根因未定（{self._slow_text(report)}），先迁移一次验证",
                **base,
            )
        return self._build(
            S.STRATEGY_OBSERVE,
            reason=f"慢节点根因未定且无迁移条件（{self._slow_text(report)}），先观测",
            **base,
        )

    def _build(self, strategy_name: str, reason: str, **base) -> Plan:
        base.pop("actions", None)  # 动作由策略注册表决定，不接受调用方覆盖
        strategy = S.get(strategy_name)
        actions = list(strategy.actions)
        return Plan(
            strategy=strategy.name,
            actions=actions,
            reason=reason,
            risk=strategy.risk,
            expected_downtime_s=strategy.downtime_s,
            params={},
            **base,
        )

    def _retries_left(self, code, ctx: JobContext) -> bool:
        budget = min(code.recurrence.max_retries, self.max_retries_cap)
        return ctx.retries_used < budget

    @staticmethod
    def _age_text(ctx: JobContext) -> str:
        if ctx.checkpoint_age_s is None:
            return "未知"
        return f"{ctx.checkpoint_age_s:.0f}s"

    @staticmethod
    def _slow_text(report: StragglerReport | None) -> str:
        if report is None:
            return "无慢节点报告"
        return f"{report.score:.2f}x, ranks={','.join(report.slow_ranks)}"
