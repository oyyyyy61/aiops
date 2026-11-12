"""统一故障码（Unified Fault Code, UFC）的数据模型。

命名规则
--------
``UFC-<DOMAIN>-<NNN>``，DOMAIN 为三位大写字母，NNN 为三位序号：

=======  ==================================================
DOMAIN   含义
=======  ==================================================
GPU      算力引擎（Xid、AI Core、GPU hang、上下文损坏）
MEM      显存 / HBM（ECC、行重映射、OOM）
LNK      片内互联（NVLink、HCCS、XGMI、PCIe）
NET      节点间网络（IB/RoCE、NCCL/HCCL 集合通信）
PWR      供电与功耗
THM      温度与散热
DRV      驱动与固件
NOD      主机 / Kubernetes 节点
JOB      训练作业语义（挂起、NaN、数据停顿、检查点）
INF      在线推理服务
OBS      观测链路（exporter、指标陈旧、指标冲突）
=======  ==================================================

序号在域内单调递增、永不复用；废弃的码保留并在 ``status: deprecated`` 中说明，
避免历史告警与工单失去指向。
"""

from __future__ import annotations

import enum
import hashlib
import time
from dataclasses import dataclass, field, replace
from typing import Any, Iterable, Mapping


class Vendor(str, enum.Enum):
    NVIDIA = "nvidia"
    AMD = "amd"
    ASCEND = "ascend"
    GENERIC = "generic"

    @classmethod
    def parse(cls, value: str | "Vendor") -> "Vendor":
        if isinstance(value, Vendor):
            return value
        key = str(value).strip().lower()
        aliases = {
            "nv": cls.NVIDIA,
            "nvidia": cls.NVIDIA,
            "cuda": cls.NVIDIA,
            "amd": cls.AMD,
            "rocm": cls.AMD,
            "mi300": cls.AMD,
            "ascend": cls.ASCEND,
            "npu": cls.ASCEND,
            "huawei": cls.ASCEND,
            "910b": cls.ASCEND,
            "generic": cls.GENERIC,
            "unknown": cls.GENERIC,
        }
        if key not in aliases:
            raise ValueError(f"未知厂商: {value!r}")
        return aliases[key]


class Severity(str, enum.Enum):
    """S1 最严重。等级决定可容忍的检测 / 处置时长（见 docs/02）。"""

    S1 = "S1"
    S2 = "S2"
    S3 = "S3"
    S4 = "S4"

    @property
    def rank(self) -> int:
        return {"S1": 1, "S2": 2, "S3": 3, "S4": 4}[self.value]

    @property
    def mttd_budget_s(self) -> int:
        """期望检测时延上限（秒）。"""

        return {"S1": 30, "S2": 120, "S3": 600, "S4": 3600}[self.value]

    @property
    def auto_action(self) -> bool:
        """S1/S2 允许自动动作，S3 允许计划性动作，S4 仅记录。"""

        return self.rank <= 2

    def __lt__(self, other: "Severity") -> bool:  # 便于排序：S1 < S2 < S3 < S4
        return self.rank < other.rank


class Scope(str, enum.Enum):
    """故障影响半径，决定隔离与迁移的最小单位。"""

    GPU = "gpu"
    NODE = "node"
    JOB = "job"
    SERVICE = "service"
    CLUSTER = "cluster"
    OBSERVABILITY = "observability"

    @property
    def isolation_target(self) -> str:
        return {
            "gpu": "gpu",
            "node": "node",
            "job": "job",
            "service": "service",
            "cluster": "cluster",
            "observability": "observability",
        }[self.value]


class Confidence(str, enum.Enum):
    """厂商码映射的可信度。

    ``verified``  厂商手册或真机日志确认过；
    ``community`` 公开 issue / 论坛 / 社区文档确认；
    ``assumed``   按语义推断，**上真机前必须核验**。
    """

    VERIFIED = "verified"
    COMMUNITY = "community"
    ASSUMED = "assumed"

    @property
    def weight(self) -> int:
        return {"verified": 3, "community": 2, "assumed": 1}[self.value]


class FaultState(str, enum.Enum):
    OPEN = "open"          # 已检测，未处置
    ACK = "ack"            # 已受理，动作执行中
    VERIFYING = "verifying"  # 动作后验证中
    RESOLVED = "resolved"
    SUPPRESSED = "suppressed"  # 演练 / 窗口期静默


# 动作词表。策略引擎只认这些名字，具体执行器在 policy/actions.py。
ACTION_EMERGENCY_CKPT = "emergency_checkpoint"
ACTION_GRACEFUL_RESTART = "graceful_restart"
ACTION_MIGRATE = "migrate_job"
ACTION_EVACUATE = "evacuate_job"
ACTION_SHRINK = "shrink_world_size"
ACTION_QUARANTINE_GPU = "quarantine_gpu"
ACTION_QUARANTINE_NODE = "quarantine_node"
ACTION_FENCE_NODE = "fence_node"
ACTION_DRAIN = "drain_instance"
ACTION_RESTART_INSTANCE = "restart_instance"
ACTION_RETRY_STEP = "retry_step"
ACTION_REDUCE_LOAD = "reduce_load"
ACTION_REDUCE_BATCH = "reduce_batch"
ACTION_ROLLBACK_DRIVER = "rollback_driver"
ACTION_NOTIFY = "notify_only"
ACTION_ESCALATE = "escalate_human"

KNOWN_ACTIONS = frozenset(
    {
        ACTION_EMERGENCY_CKPT,
        ACTION_GRACEFUL_RESTART,
        ACTION_MIGRATE,
        ACTION_EVACUATE,
        ACTION_SHRINK,
        ACTION_QUARANTINE_GPU,
        ACTION_QUARANTINE_NODE,
        ACTION_FENCE_NODE,
        ACTION_DRAIN,
        ACTION_RESTART_INSTANCE,
        ACTION_RETRY_STEP,
        ACTION_REDUCE_LOAD,
        ACTION_REDUCE_BATCH,
        ACTION_ROLLBACK_DRIVER,
        ACTION_NOTIFY,
        ACTION_ESCALATE,
    }
)

# 验证运算符。恢复动作必须绑定可检查的后验条件（见 docs/02 §5）。
VERIFY_CHECKS = frozenset(
    {
        "step_advance",   # 训练步数继续推进
        "ckpt_advance",   # 检查点持续刷新
        "bench_pass",     # 微基准达标
        "ecc_clear",      # 无新增不可纠正 ECC
        "fault_absent",   # 窗口内故障码不再出现
        "svc_healthy",    # 健康检查连续通过
        "latency_slo",    # p99 回到 SLO 内
        "node_ready",     # 节点重新 Ready
    }
)

CHECKPOINT_POLICIES = frozenset({"none", "periodic", "emergency", "both"})


@dataclass(frozen=True)
class Target:
    """故障/动作作用对象。空串表示该维度不适用。"""

    node: str = ""
    gpu: str = ""
    job: str = ""
    service: str = ""
    vendor: Vendor = Vendor.GENERIC

    def key(self) -> str:
        parts = [
            f"{k}={v}"
            for k, v in (
                ("node", self.node),
                ("gpu", self.gpu),
                ("job", self.job),
                ("svc", self.service),
            )
            if v
        ]
        return ",".join(parts) or "unknown"

    def with_gpu(self, gpu: str) -> "Target":
        return replace(self, gpu=gpu)

    def as_dict(self) -> dict[str, str]:
        return {"node": self.node, "gpu": self.gpu, "job": self.job, "service": self.service}


@dataclass(frozen=True)
class SignalRule:
    """指标型检测规则。事件型故障（厂商码）不需要写这里。"""

    metric: str
    op: str                 # gt / ge / lt / le / eq / ne / rate_gt
    value: float
    for_samples: int = 1    # 连续 N 个采样点成立才判故障（去抖）
    clear_samples: int = 2  # 连续 N 个采样点不成立才判定恢复
    note: str = ""

    def evaluate(self, observed: float, previous: float | None = None) -> bool:
        op = self.op
        if op == "gt":
            return observed > self.value
        if op == "ge":
            return observed >= self.value
        if op == "lt":
            return observed < self.value
        if op == "le":
            return observed <= self.value
        if op == "eq":
            return observed == self.value
        if op == "ne":
            return observed != self.value
        if op == "rate_gt":
            if previous is None:
                return False
            return (observed - previous) > self.value
        raise ValueError(f"不支持的比较符: {op}")


@dataclass(frozen=True)
class VendorMapping:
    vendor: Vendor
    codes: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()
    confidence: Confidence = Confidence.ASSUMED
    family: str = ""   # 例如 "Tesla/Hopper"、"MI300X"、"910B"
    note: str = ""


@dataclass(frozen=True)
class Recurrence:
    """复发处理：重试几次后升级，避免无限重启放大故障。"""

    max_retries: int = 1
    quarantine_after: int = 2
    backoff_s: float = 5.0


@dataclass(frozen=True)
class FaultCode:
    id: str
    name: str
    domain: str
    severity: Severity
    scope: Scope
    summary: str
    actions: tuple[str, ...]
    checkpoint: str = "periodic"
    verify: tuple[str, ...] = ()
    runbook: tuple[str, ...] = ()
    recurrence: Recurrence = Recurrence()
    signals: tuple[SignalRule, ...] = ()
    vendor_map: tuple[VendorMapping, ...] = ()
    tags: tuple[str, ...] = ()
    status: str = "active"

    def __post_init__(self) -> None:
        if not self.id.startswith("UFC-"):
            raise ValueError(f"故障码必须以 UFC- 开头: {self.id}")
        parts = self.id.split("-")
        if len(parts) != 3 or len(parts[1]) != 3 or len(parts[2]) != 3:
            raise ValueError(f"故障码格式应为 UFC-DOM-NNN: {self.id}")
        if parts[1] != self.domain:
            raise ValueError(f"{self.id} 的 domain 与编号前缀不一致: {self.domain}")
        unknown = set(self.actions) - KNOWN_ACTIONS
        if unknown:
            raise ValueError(f"{self.id} 含未登记动作: {sorted(unknown)}")
        if not self.actions:
            raise ValueError(f"{self.id} 必须至少声明一个动作")
        bad_verify = set(self.verify) - VERIFY_CHECKS
        if bad_verify:
            raise ValueError(f"{self.id} 含未登记验证项: {sorted(bad_verify)}")
        if self.checkpoint not in CHECKPOINT_POLICIES:
            raise ValueError(f"{self.id} checkpoint 策略非法: {self.checkpoint}")
        if self.severity.auto_action and not self.verify:
            raise ValueError(f"{self.id} 为 {self.severity.value} 自动动作，必须绑定 verify 后验条件")

    # ---- 查询辅助 ----

    @property
    def vendor_labels(self) -> tuple[str, ...]:
        return tuple(m.vendor.value for m in self.vendor_map)

    @property
    def min_confidence(self) -> Confidence:
        if not self.vendor_map:
            return Confidence.ASSUMED
        return min((m.confidence for m in self.vendor_map), key=lambda c: c.weight)

    def label(self) -> str:
        return f"{self.id} {self.name}"

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "domain": self.domain,
            "severity": self.severity.value,
            "scope": self.scope.value,
            "summary": self.summary,
            "actions": list(self.actions),
            "checkpoint": self.checkpoint,
            "verify": list(self.verify),
            "runbook": list(self.runbook),
            "recurrence": {
                "max_retries": self.recurrence.max_retries,
                "quarantine_after": self.recurrence.quarantine_after,
                "backoff_s": self.recurrence.backoff_s,
            },
            "signals": [
                {
                    "metric": s.metric,
                    "op": s.op,
                    "value": s.value,
                    "for_samples": s.for_samples,
                    "clear_samples": s.clear_samples,
                }
                for s in self.signals
            ],
            "vendor_map": [
                {
                    "vendor": m.vendor.value,
                    "codes": list(m.codes),
                    "patterns": list(m.patterns),
                    "confidence": m.confidence.value,
                    "family": m.family,
                    "note": m.note,
                }
                for m in self.vendor_map
            ],
            "tags": list(self.tags),
        }


@dataclass
class VendorEvent:
    """原始厂商事件（一条 Xid、一条 amdgpu 日志、一条 npu-smi 告警）。"""

    vendor: Vendor
    code: str = ""
    text: str = ""
    component: str = ""
    ts: float = field(default_factory=time.time)
    raw: Mapping[str, Any] = field(default_factory=dict)

    def fingerprint(self) -> str:
        seed = f"{self.vendor.value}|{self.code}|{self.component}|{self.text[:120]}"
        return hashlib.sha1(seed.encode()).hexdigest()[:12]


@dataclass
class Fault:
    """一条运行期故障实例（UFC 码 + 作用对象 + 证据）。"""

    code: FaultCode
    target: Target
    detected_at: float
    onset_at: float
    evidence: list[str] = field(default_factory=list)
    state: FaultState = FaultState.OPEN
    resolved_at: float | None = None
    attempts: int = 0
    last_action: str = ""
    detection_latency_s: float = 0.0

    @property
    def fingerprint(self) -> str:
        return f"{self.code.id}|{self.target.key()}"

    @property
    def severity(self) -> Severity:
        return self.code.severity

    def acknowledge(self) -> None:
        self.state = FaultState.ACK

    def resolve(self, ts: float | None = None) -> None:
        self.state = FaultState.RESOLVED
        self.resolved_at = ts if ts is not None else time.time()

    def as_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.id,
            "name": self.code.name,
            "severity": self.severity.value,
            "scope": self.code.scope.value,
            "target": self.target.key(),
            "target_fields": self.target.as_dict(),
            "onset_at": round(self.onset_at, 3),
            "detected_at": round(self.detected_at, 3),
            "detection_latency_s": round(self.detection_latency_s, 3),
            "state": self.state.value,
            "evidence": self.evidence,
            "attempts": self.attempts,
            "last_action": self.last_action,
        }


def fingerprint(code_id: str, target: Target) -> str:
    return f"{code_id}|{target.key()}"


def sort_codes(codes: Iterable[FaultCode]) -> list[FaultCode]:
    return sorted(codes, key=lambda c: (c.domain, c.id))
