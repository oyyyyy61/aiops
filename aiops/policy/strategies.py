"""恢复策略注册表。

一个"策略"= 一组有序动作 + 前置条件 + 风险 + 预期停机时间。
策略本身不执行任何东西，只描述"要做什么、需要什么、代价多大"，
由 supervisor 按计划执行并验证后验条件。

当前实现 6 种可自动执行的策略（题目要求 ≥2）：

===================================  ==================================================
策略                                  适用场景
===================================  ==================================================
``restart_in_place``                  可恢复故障、有检查点、同节点同卡直接重启
``migrate``                           卡/节点硬件可疑、有备机，迁走后再处置
``evacuate``                          节点级致命故障（掉卡、PSU、宕机），整机清空
``shrink``                            慢节点或坏卡无备机时，去掉该 rank 缩小规模续跑
``drain_restart``                     推理实例摘流 -> 重启 -> 回填，保证在线请求不被打断
``degrade``                           容量类问题（OOM、队列积压），降载不重启
===================================  ==================================================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

STRATEGY_RESTART = "restart_in_place"
STRATEGY_MIGRATE = "migrate"
STRATEGY_EVACUATE = "evacuate"
STRATEGY_SHRINK = "shrink"
STRATEGY_DRAIN = "drain_restart"
STRATEGY_DEGRADE = "degrade"
STRATEGY_OBSERVE = "observe"
STRATEGY_FAIL_FAST = "fail_fast"


@dataclass(frozen=True)
class Strategy:
    name: str
    label: str
    actions: tuple[str, ...]
    requires_checkpoint: bool = False
    requires_spare: bool = False
    risk: str = "low"          # low / medium / high
    downtime_s: float = 30.0   # 预期停机（不含加载检查点的时间）
    verifies: tuple[str, ...] = ()
    description: str = ""


REGISTRY: dict[str, Strategy] = {
    s.name: s
    for s in (
        Strategy(
            name=STRATEGY_RESTART,
            label="原地重启",
            actions=("emergency_checkpoint", "graceful_restart"),
            requires_checkpoint=True,
            risk="low",
            downtime_s=45.0,
            verifies=("step_advance", "fault_absent"),
            description="同一节点同一组卡上重启进程，从最近检查点续训；不改变拓扑。",
        ),
        Strategy(
            name=STRATEGY_MIGRATE,
            label="迁移重启",
            actions=("emergency_checkpoint", "migrate_job", "quarantine_gpu"),
            requires_checkpoint=True,
            requires_spare=True,
            risk="medium",
            downtime_s=120.0,
            verifies=("step_advance", "bench_pass"),
            description="把作业迁到备用节点/健康卡，原设备隔离；适合卡级硬件可疑。",
        ),
        Strategy(
            name=STRATEGY_EVACUATE,
            label="节点疏散",
            actions=("emergency_checkpoint", "evacuate_job", "quarantine_node", "escalate_human"),
            requires_checkpoint=True,
            requires_spare=True,
            risk="high",
            downtime_s=180.0,
            verifies=("node_ready", "fault_absent"),
            description="节点级致命故障：整机退出调度池并报修，作业迁走。",
        ),
        Strategy(
            name=STRATEGY_SHRINK,
            label="缩容续跑",
            actions=("emergency_checkpoint", "shrink_world_size"),
            requires_checkpoint=True,
            risk="medium",
            downtime_s=150.0,
            verifies=("step_advance", "bench_pass"),
            description="去掉坏 rank、缩小 world size 继续训练（同步调整全局 batch 与学习率）。",
        ),
        Strategy(
            name=STRATEGY_DRAIN,
            label="摘流重启",
            actions=("drain_instance", "restart_instance"),
            risk="low",
            downtime_s=30.0,
            verifies=("svc_healthy", "latency_slo"),
            description="推理实例先从负载均衡摘除、等待在途请求结束，再重启并回填。",
        ),
        Strategy(
            name=STRATEGY_DEGRADE,
            label="降载续跑",
            actions=("reduce_load",),
            risk="low",
            downtime_s=0.0,
            verifies=("latency_slo",),
            description="容量类问题：降 batch/并发、缩小序列长度，不重启进程。",
        ),
        Strategy(
            name=STRATEGY_OBSERVE,
            label="只观测",
            actions=("notify_only",),
            risk="low",
            downtime_s=0.0,
            verifies=(),
            description="证据不足或等级低：只记录与告警，不做拓扑变更。",
        ),
        Strategy(
            name=STRATEGY_FAIL_FAST,
            label="失败并升级",
            actions=("escalate_human",),
            risk="low",
            downtime_s=0.0,
            verifies=(),
            description="没有检查点/备机/重试预算时，停止自动动作并升级人工，避免盲目重启。",
        ),
    )
}

# 策略强度阶梯：重试耗尽后逐级升级
ESCALATION = {
    STRATEGY_OBSERVE: STRATEGY_RESTART,
    STRATEGY_DEGRADE: STRATEGY_OBSERVE,
    STRATEGY_RESTART: STRATEGY_MIGRATE,
    STRATEGY_MIGRATE: STRATEGY_EVACUATE,
    STRATEGY_SHRINK: STRATEGY_OBSERVE,
    STRATEGY_EVACUATE: STRATEGY_FAIL_FAST,
    STRATEGY_DRAIN: STRATEGY_FAIL_FAST,
    STRATEGY_FAIL_FAST: STRATEGY_FAIL_FAST,
}


def get(name: str) -> Strategy:
    return REGISTRY[name]


@dataclass
class Plan:
    """一次恢复决策的输出，可审计、可回放。"""

    strategy: str
    fault_code: str
    fault_name: str
    severity: str
    target: str
    reason: str
    actions: list[str]
    params: dict = field(default_factory=dict)
    risk: str = "low"
    expected_downtime_s: float = 0.0
    degraded_by_telemetry: bool = False
    escalation_step: int = 0

    @property
    def is_noop(self) -> bool:
        return self.strategy in (STRATEGY_OBSERVE, STRATEGY_FAIL_FAST)

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "fault_code": self.fault_code,
            "fault_name": self.fault_name,
            "severity": self.severity,
            "target": self.target,
            "reason": self.reason,
            "actions": self.actions,
            "params": self.params,
            "risk": self.risk,
            "expected_downtime_s": self.expected_downtime_s,
            "degraded_by_telemetry": self.degraded_by_telemetry,
            "escalation_step": self.escalation_step,
        }
