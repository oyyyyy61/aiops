"""作业监督器：把「检测 -> 决策 -> 保存 -> 重启/迁移 -> 验证」串成闭环。

它是整个系统的执行面，直接持有子进程。设计约束：

* **跨平台**：Linux 用 SIGTERM 触发优雅保存；Windows / 演练场景用控制文件，
  避免依赖 POSIX 信号（本地只有 Windows + 消费级卡）。
* **可回放**：每一步（检测、决策、动作、验证）都写进 timeline，事后能回答
  "为什么在第 137 秒迁移了作业"。
* **可度量**：记录 MTTD（检测时延）、MTTR（恢复时延）、RPO（丢失步数）、
  动作耗时，直接产出实验表格。
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from ..collect.base import Snapshot
from ..detect.rules import Detector
from ..detect.straggler import (
    BenchMonitor,
    StepTimeMonitor,
    StragglerReport,
    classify_root_cause,
)
from ..detect.telemetry import ActionGate, TelemetryMonitor
from ..faultcode.models import Fault, FaultState, Target
from ..policy import strategies as S
from ..policy.engine import JobContext, PolicyEngine
from .checkpoint import CheckpointMeta, CheckpointStore

PHASE_RUNNING = "RUNNING"
PHASE_FAULTED = "FAULTED"
PHASE_SAVING = "SAVING"
PHASE_RECOVERING = "RECOVERING"
PHASE_VERIFYING = "VERIFYING"
PHASE_DONE = "DONE"
PHASE_FAILED = "FAILED"


@dataclass
class JobSpec:
    name: str
    argv: list[str]
    kind: str = "train"
    node: str = ""
    world_size: int = 1
    env: Mapping[str, str] = field(default_factory=dict)
    cwd: str | None = None
    ckpt_dir: str = ""
    control_dir: str = ""
    stop_mode: str = "control"   # control | signal
    grace_s: float = 20.0
    max_steps: int | None = None
    spare_nodes: list[str] = field(default_factory=list)
    world_size_flag: str = "--world-size"

    def spawn_argv(self, *, resume: bool, world_size: int | None = None, node: str | None = None,
                   exclude_ranks: list[str] | None = None) -> list[str]:
        """按策略改写启动参数。真实平台里这里换成 torchrun/DeepSpeed 的 argv。"""

        argv = list(self.argv)
        if resume:
            argv.append("--resume")
        flag = self.world_size_flag
        if world_size is not None and world_size != self.world_size:
            if flag in argv:
                argv[argv.index(flag) + 1] = str(world_size)
            else:
                argv += [flag, str(world_size)]
        if node and node != self.node:
            if "--node" in argv:
                argv[argv.index("--node") + 1] = node
            else:
                argv += ["--node", node]
        for rank in exclude_ranks or []:
            argv += ["--exclude-rank", str(rank)]
        return argv


@dataclass
class JobStatus:
    running: bool = False
    rc: int | None = None
    step: int = 0
    step_index: int = 0
    step_time_s: float = 0.0
    step_times: dict[str, float] = field(default_factory=dict)
    progress_age_s: float = 0.0
    state: str = "unknown"     # running / saving / done / failed
    node: str = ""
    world_size: int = 1
    loss: float = 0.0
    extra: dict = field(default_factory=dict)


@dataclass
class Probe:
    snapshot: Snapshot
    status: JobStatus


@dataclass
class RecoveryRecord:
    fault_code: str
    fault_name: str
    severity: str
    target: str
    strategy: str
    reason: str
    failure_step: int
    ckpt_step: int | None
    rpo_steps: int
    pre_fault_ckpt_step: int | None = None
    onset_ts: float = 0.0
    detected_ts: float = 0.0
    action_ts: float = 0.0
    resumed_ts: float = 0.0
    verified_ts: float = 0.0
    verified: bool = False
    attempts: int = 1
    escalated: bool = False
    target_node: str = ""

    @property
    def mttd_s(self) -> float:
        """检测时延：从故障发生（onset）到被检测出来的时间。"""

        return max(0.0, self.detected_ts - self.onset_ts) if self.detected_ts else 0.0

    @property
    def mttr_s(self) -> float:
        end = self.verified_ts or self.resumed_ts
        return max(0.0, end - self.detected_ts)

    @property
    def rpo_without_emergency(self) -> int:
        """如果只有周期检查点（没有紧急保存），会丢多少步。"""

        if self.pre_fault_ckpt_step is None or self.failure_step <= 0:
            return -1
        return max(0, self.failure_step - self.pre_fault_ckpt_step)

    @property
    def downtime_s(self) -> float:
        return max(0.0, self.resumed_ts - self.action_ts)

    def as_dict(self) -> dict:
        return {
            "fault_code": self.fault_code,
            "fault_name": self.fault_name,
            "severity": self.severity,
            "target": self.target,
            "strategy": self.strategy,
            "reason": self.reason,
            "failure_step": self.failure_step,
            "ckpt_step": self.ckpt_step,
            "rpo_steps": self.rpo_steps,
            "pre_fault_ckpt_step": self.pre_fault_ckpt_step,
            "rpo_without_emergency": self.rpo_without_emergency,
            "detected_ts": round(self.detected_ts, 3),
            "action_ts": round(self.action_ts, 3),
            "resumed_ts": round(self.resumed_ts, 3),
            "verified_ts": round(self.verified_ts, 3),
            "verified": self.verified,
            "attempts": self.attempts,
            "escalated": self.escalated,
            "target_node": self.target_node,
            "downtime_s": round(self.downtime_s, 3),
            "mttr_s": round(self.mttr_s, 3),
        }


@dataclass
class RunResult:
    job: str
    ok: bool
    phase: str
    steps_completed: int
    recoveries: list[RecoveryRecord] = field(default_factory=list)
    faults: list[dict] = field(default_factory=list)
    timeline: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def recovery_success_rate(self) -> float:
        if not self.recoveries:
            return 1.0
        return sum(1 for r in self.recoveries if r.verified) / len(self.recoveries)

    def summary(self) -> dict:
        verified = [r for r in self.recoveries if r.verified]
        return {
            "job": self.job,
            "ok": self.ok,
            "phase": self.phase,
            "steps_completed": self.steps_completed,
            "faults": len(self.faults),
            "recoveries": len(self.recoveries),
            "recovered": len(verified),
            "recovery_success_rate": round(self.recovery_success_rate, 3),
            "mttd_avg_s": round(sum(r.mttd_s for r in verified) / len(verified), 2) if verified else None,
            "mttr_avg_s": round(sum(r.mttr_s for r in verified) / len(verified), 2) if verified else None,
            "downtime_avg_s": round(sum(r.downtime_s for r in verified) / len(verified), 2) if verified else None,
            "rpo_steps_avg": round(sum(r.rpo_steps for r in verified) / len(verified), 1) if verified else None,
            "rpo_steps_max": max((r.rpo_steps for r in verified), default=0),
            "rpo_without_emergency_avg": (
                round(sum(r.rpo_without_emergency for r in verified if r.rpo_without_emergency >= 0)
                      / max(1, len([r for r in verified if r.rpo_without_emergency >= 0])), 1)
                if any(r.rpo_without_emergency >= 0 for r in verified) else None
            ),
            "escalations": sum(1 for r in self.recoveries if r.escalated),
            "wall_s": round(self.wall_s, 2),
        }


class Supervisor:
    """管理一个作业进程的完整生命周期。"""

    def __init__(
        self,
        spec: JobSpec,
        store: CheckpointStore,
        detector: Detector,
        policy: PolicyEngine,
        probe: Callable[[], Probe],
        telemetry: TelemetryMonitor | None = None,
        gate: ActionGate | None = None,
        poll_s: float = 0.2,
        verify_steps: int = 5,
        verify_window_s: float = 8.0,
        clock: Callable[[], float] = time.time,
        bench: BenchMonitor | None = None,
    ):
        self.spec = spec
        self.store = store
        self.detector = detector
        self.policy = policy
        self.probe = probe
        self.monitor = telemetry or TelemetryMonitor(expected_interval_s=1.0)
        self.gate = gate or ActionGate(self.monitor)
        self.poll_s = poll_s
        self.verify_steps = verify_steps
        self.verify_window_s = verify_window_s
        self.clock = clock
        self.bench = bench

        self.proc: subprocess.Popen | None = None
        self.phase = PHASE_RUNNING
        self.attempt = 0
        self.timeline: list[dict] = []
        self.records: list[RecoveryRecord] = []
        self.handled: dict[str, float] = {}       # fingerprint -> ts，冷却用
        self._cooldown_s = 5.0
        self._current: RecoveryRecord | None = None
        self._current_plan: S.Plan | None = None
        self._recovery_ckpt: CheckpointMeta | None = None
        self._recovery_started = 0.0
        self._exclude: list[str] = []
        self._world_size = spec.world_size
        self._node = spec.node
        self._straggler = StepTimeMonitor(spec.name, window=20)
        self._straggler_report: StragglerReport | None = None
        self._straggler_cause = "unknown"
        self._last_seen_step = -1
        self._faults_log: list[dict] = []
        self._stop_requested = False
        self._current_fault: Fault | None = None
        self._failure_step: int = 0
        self._recovery_step: int = 0
        self._started_at: float = 0.0

    # ---------- 日志 ----------

    def _log(self, kind: str, detail: str = "", **extra: Any) -> None:
        entry = {"ts": round(self.clock(), 3), "kind": kind, "detail": detail, **extra}
        self.timeline.append(entry)

    # ---------- 进程控制 ----------

    def start(self, *, resume: bool = False, node: str | None = None, world_size: int | None = None) -> None:
        if node:
            self._node = node
        if world_size is not None:
            self._world_size = world_size
        argv = self.spec.spawn_argv(
            resume=resume, world_size=self._world_size, node=self._node, exclude_ranks=self._exclude
        )
        env = dict(os.environ)
        env.update({str(k): str(v) for k, v in self.spec.env.items()})
        env["AIOPS_JOB"] = self.spec.name
        env["AIOPS_ATTEMPT"] = str(self.attempt)
        env["AIOPS_RESUME"] = "1" if resume else "0"
        env["AIOPS_NODE"] = self._node
        self.attempt += 1
        self._started_at = self.clock()
        self._clear_control()
        self.proc = subprocess.Popen(
            argv,
            cwd=self.spec.cwd,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.phase = PHASE_RUNNING
        self._log("job_start", f"attempt={self.attempt}", pid=self.proc.pid, node=self._node,
                  world_size=self._world_size, resume=resume)

    def _clear_control(self) -> None:
        if not self.spec.control_dir:
            return
        ctl = Path(self.spec.control_dir)
        if not ctl.exists():
            return
        for name in ("graceful", "ckpt_ready", "nan", "force_exit"):
            (ctl / name).unlink(missing_ok=True)
        for pattern in ("die.*", "hang.*", "throttle.*"):
            for path in ctl.glob(pattern):
                path.unlink(missing_ok=True)

    def _signal_graceful(self) -> None:
        """请求优雅保存：Linux 发 SIGTERM，Windows/演练写控制文件。"""

        if self.spec.stop_mode == "signal" and os.name != "nt" and self.proc:
            try:
                self.proc.send_signal(signal.SIGTERM)
            except OSError:
                pass
            return
        if self.spec.control_dir:
            _write_atomic(Path(self.spec.control_dir) / "graceful", "1")

    def stop(self, timeout: float | None = None) -> int | None:
        """优雅停止，超时后强杀。返回退出码。"""

        if self.proc is None:
            return None
        timeout = self.spec.grace_s if timeout is None else timeout
        if self.proc.poll() is None:
            self._signal_graceful()
            deadline = self.clock() + timeout
            while self.clock() < deadline and self.proc.poll() is None:
                time.sleep(0.05)
        if self.proc.poll() is None:
            if self.spec.control_dir:
                try:
                    (Path(self.spec.control_dir) / "force_exit").write_text("1", encoding="utf-8")
                except OSError:
                    pass
                time.sleep(1.0)
        if self.proc.poll() is None:
            self._log("force_kill", f"grace={timeout}s 未退出")
            self.proc.kill()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        rc = self.proc.returncode
        self._log("job_stop", f"rc={rc}")
        self.proc = None
        return rc

    # ---------- 主循环 ----------

    def run(self, max_seconds: float = 900.0, target_step: int | None = None) -> RunResult:
        start = self.clock()
        if self.proc is None:
            self.start()
        last_step = 0
        progress_ts = start
        while self.clock() - start < max_seconds:
            time.sleep(self.poll_s)
            probe = self.probe()
            snap, status = probe.snapshot, probe.status
            self.monitor.observe(snap)
            if status.step > self._last_seen_step:
                self._straggler.update_many(status.step_times)
                self._last_seen_step = status.step
            self._refresh_straggler(status)

            det = self.detector.observe(snap, now=self.clock())
            for fault in det.new:
                self._on_fault(fault, status)

            # 进程异常退出但没有故障事件 -> 合成一条作业级故障
            if self.proc is not None and self.proc.poll() is not None:
                rc = self.proc.returncode
                self._log("job_exit", f"rc={rc}")
                self.proc = None
                if self.phase in (PHASE_RUNNING, PHASE_FAULTED) and rc not in (0,):
                    self._synthesize_fault("UFC-JOB-001", f"作业异常退出 rc={rc}", status)
                elif rc == 0:
                    self.phase = PHASE_DONE
                    break

            if self._current_plan is not None:
                self._execute_plan(self._current_plan, status)
                self._current_plan = None
                continue

            if self.phase == PHASE_RECOVERING and self._current is not None:
                self._verify(status)

            if self.phase == PHASE_FAILED:
                self._log("run_failed", "进入失败态，停止自动动作")
                break
            if status.step > last_step:
                last_step = status.step
                progress_ts = self.clock()
            if target_step is not None and status.step >= target_step:
                self._log("target_reached", f"step={status.step}")
                break
            # 刚重启的瞬间，探针可能还停在上一次尝试的"已完成"心跳上；
            # 给新进程一个短暂的启动窗口，避免把旧状态当成新结果。
            if status.state == "done" and self.clock() - self._started_at > 1.0:
                self.phase = PHASE_DONE
                break

        if self.proc is not None:
            self.stop()

        wall = self.clock() - start
        result = RunResult(
            job=self.spec.name,
            ok=self.phase == PHASE_DONE,
            phase=self.phase,
            steps_completed=max(last_step, self._last_ckpt_step()),
            recoveries=self.records,
            faults=self._faults_log,
            timeline=self.timeline,
            wall_s=wall,
        )
        return result

    # ---------- 故障与决策 ----------

    def _on_fault(self, fault: Fault, status: JobStatus) -> None:
        fp = fault.fingerprint
        if fp in self.handled and self.clock() - self.handled[fp] < self._cooldown_s:
            return
        self.handled[fp] = self.clock()
        self._faults_log.append(fault.as_dict())
        self._log(
            "fault_detected",
            f"{fault.code.id} {fault.code.name}",
            severity=fault.severity.value,
            target=fault.target.key(),
            evidence=fault.evidence[:3],
            detection_latency_s=round(fault.detection_latency_s, 2),
        )
        ctx = self._context(status)
        plan = self.policy.decide(fault, ctx)
        self._log(
            "decision",
            f"{plan.strategy}: {plan.reason}",
            strategy=plan.strategy,
            actions=plan.actions,
            risk=plan.risk,
            degraded_by_telemetry=plan.degraded_by_telemetry,
        )
        if plan.is_noop:
            if plan.strategy == S.STRATEGY_FAIL_FAST:
                self.phase = PHASE_FAILED
            return
        # 每个故障只执行一次；执行后进入恢复流程
        if self.phase in (PHASE_FAULTED, PHASE_SAVING, PHASE_RECOVERING) and self._current is not None:
            return
        self._current_plan = plan
        self._current_fault = fault
        self._failure_step = status.step

    def _context(self, status: JobStatus) -> JobContext:
        latest = self.store.latest()
        return JobContext(
            job=self.spec.name,
            kind=self.spec.kind,
            node=self._node,
            world_size=self._world_size,
            checkpoint_available=latest is not None,
            checkpoint_age_s=latest.age_s(self.clock()) if latest else None,
            retries_used=self.attempt - 1,
            spare_nodes=max(0, len(self.spec.spare_nodes)),
            telemetry=self.monitor.evaluate(self.clock()),
            straggler=self._straggler_report,
            straggler_cause=self._straggler_cause,
        )

    def _refresh_straggler(self, status: JobStatus) -> None:
        report = self._straggler.evaluate()
        if report is None:
            self._straggler_report = None
            self._straggler_cause = "unknown"
            return
        self._straggler_report = report
        # 微基准证据：慢 rank 所在节点是否也出现在 bench 慢设备里
        self._straggler_cause = classify_root_cause(
            report,
            self.bench,
            node_of={r: status.node or self._node for r in report.slow_ranks},
        )
        if report.slow_ranks:
            code = self.detector.catalog.get("UFC-JOB-007")
            fault = self._straggler.to_fault(report, self.detector.catalog,
                                            node_of={r: status.node or self._node for r in report.slow_ranks})
            # 慢节点没有精确的"故障时刻"，用持续时长 * 中位步耗时反推起点
            now = self.clock()
            fault.detected_at = now
            fault.onset_at = now - report.duration * report.median_step_s
            fault.detection_latency_s = now - fault.onset_at
            fp = fault.fingerprint
            if fp not in self.handled:
                self._log("straggler", f"score={report.score:.2f} ranks={report.slow_ranks}",
                          cause=self._straggler_cause)
                self._on_fault(fault, status)

    def _synthesize_fault(self, code_id: str, detail: str, status: JobStatus) -> None:
        code = self.detector.catalog.get(code_id)
        fault = Fault(
            code=code,
            target=Target(node=self._node, job=self.spec.name),
            detected_at=self.clock(),
            onset_at=self.clock(),
            evidence=[detail],
        )
        self._faults_log.append(fault.as_dict())
        self._log("fault_detected", f"{code.id} {code.name}（{detail}）", severity=code.severity.value)
        ctx = self._context(status)
        plan = self.policy.decide(fault, ctx)
        self._log("decision", f"{plan.strategy}: {plan.reason}", strategy=plan.strategy, actions=plan.actions)
        if plan.is_noop:
            if plan.strategy == S.STRATEGY_FAIL_FAST:
                self._log("fail_fast", plan.reason)
                self.phase = PHASE_FAILED
                self.stop()
            return
        self._current_plan = plan
        self._current_fault = fault
        self._failure_step = status.step

    # ---------- 执行 ----------

    def _execute_plan(self, plan: S.Plan, status: JobStatus) -> None:
        fault = self._current_fault
        failure_step = self._failure_step
        self.phase = PHASE_FAULTED
        action_ts = self.clock()
        self._log("action_start", plan.strategy, actions=plan.actions)
        if fault is not None:
            # 已进入处置流程，先关闭该故障实例；复发会生成新实例再决策
            self.detector.resolve(fault.fingerprint)

        # 0) 只读/降载类动作不打断进程
        if plan.is_noop or plan.strategy == S.STRATEGY_DEGRADE:
            record = RecoveryRecord(
                fault_code=fault.code.id if fault else plan.fault_code,
                fault_name=fault.code.name if fault else plan.fault_name,
                severity=fault.severity.value if fault else plan.severity,
                target=fault.target.key() if fault else plan.target,
                strategy=plan.strategy,
                reason=plan.reason,
                failure_step=failure_step,
                ckpt_step=self._last_ckpt_step() or None,
                rpo_steps=0,
                onset_ts=fault.onset_at if fault else action_ts,
                detected_ts=fault.detected_at if fault else action_ts,
                action_ts=action_ts,
                resumed_ts=action_ts,
                verified_ts=self.clock(),
                verified=True,
                attempts=self.attempt,
            )
            self.records.append(record)
            self._log("degrade", "已降载，进程继续运行")
            self.phase = PHASE_RUNNING
            return

        # 1) 保存检查点（带超时窗口）
        ckpt_step = None
        before = self.store.latest()
        if "emergency_checkpoint" in plan.actions:
            self.phase = PHASE_SAVING
            self._signal_graceful()
            self._log("checkpoint_requested", "已请求优雅保存")
            self.stop(timeout=self.spec.grace_s)
            after = self.store.latest()
            if after is not None:
                ckpt_step = after.step
                improved = before is None or after.step > before.step
                self._log("checkpoint_saved", f"step={after.step}", improved=improved)
            else:
                self._log("checkpoint_missing", "没有任何检查点，无法恢复进度")
        else:
            self.stop(timeout=5.0)
        ckpt_step = ckpt_step if ckpt_step is not None else (
            self.store.latest().step if self.store.latest() else None
        )

        # 2) 拓扑变化
        target_node = self._node
        if plan.strategy == S.STRATEGY_MIGRATE or plan.strategy == S.STRATEGY_EVACUATE:
            spare = self.spec.spare_nodes
            if spare:
                target_node = spare[self._world_size % len(spare)]
                self._log("migrate", f"{self._node} -> {target_node}")
            else:
                self._log("migrate_skipped", "无备用节点，退回原地重启")
        elif plan.strategy == S.STRATEGY_SHRINK:
            if self._world_size > 1:
                self._world_size -= 1
                self._log("shrink", f"world_size -> {self._world_size}")
        elif plan.strategy == S.STRATEGY_DRAIN:
            self._log("drain", "负载均衡已摘除实例")

        # 3) 记录本次恢复
        record = RecoveryRecord(
            fault_code=fault.code.id if fault else plan.fault_code,
            fault_name=fault.code.name if fault else plan.fault_name,
            severity=fault.severity.value if fault else plan.severity,
            target=fault.target.key() if fault else plan.target,
            strategy=plan.strategy,
            reason=plan.reason,
            failure_step=failure_step,
            ckpt_step=ckpt_step,
            rpo_steps=max(0, failure_step - ckpt_step) if ckpt_step is not None else -1,
            pre_fault_ckpt_step=before.step if before else None,
            onset_ts=fault.onset_at if fault else action_ts,
            detected_ts=fault.detected_at if fault else action_ts,
            action_ts=action_ts,
            attempts=self.attempt,
            target_node=target_node,
        )

        needs_restart = plan.strategy in (
            S.STRATEGY_RESTART, S.STRATEGY_MIGRATE, S.STRATEGY_EVACUATE,
            S.STRATEGY_SHRINK, S.STRATEGY_DRAIN,
        )
        if not needs_restart:
            # 降载类：不重启，直接恢复
            self._log("degrade", "已降载，不重启进程")
            record.resumed_ts = self.clock()
            record.verified = True
            record.verified_ts = self.clock()
            self.records.append(record)
            self.phase = PHASE_RUNNING
            self.start(resume=False, node=self._node, world_size=self._world_size)
            return

        if plan.strategy == S.STRATEGY_DRAIN:
            self._log("instance_restart", "推理实例重启并回填")
        # 4) 重启（从检查点续训）
        self._log("relaunch", f"node={target_node} world_size={self._world_size} resume=True")
        self.start(resume=True, node=target_node, world_size=self._world_size)
        record.resumed_ts = self.clock()
        self._current = record
        self._recovery_ckpt = self.store.latest()
        self._recovery_started = self.clock()
        self._recovery_step = ckpt_step or 0
        self.phase = PHASE_RECOVERING
        self._log("recovering", f"等待 step > {self._recovery_step + self.verify_steps}")

    # ---------- 验证 ----------

    def _verify(self, status: JobStatus) -> None:
        record = self._current
        if record is None:
            return
        ckpt = self._recovery_ckpt
        target_step = (ckpt.step if ckpt else 0) + self.verify_steps
        elapsed = self.clock() - self._recovery_started
        fault_absent = not any(
            f.code.id == record.fault_code for f in self.detector.active.values()
        )
        if status.step > target_step and fault_absent:
            record.verified = True
            record.verified_ts = self.clock()
            self._log(
                "recovery_verified",
                f"strategy={record.strategy} step={status.step} mttr={record.mttr_s:.1f}s "
                f"rpo={record.rpo_steps}",
                step=status.step,
                mttr_s=round(record.mttr_s, 2),
                rpo_steps=record.rpo_steps,
            )
            self.records.append(record)
            self._current = None
            self.phase = PHASE_RUNNING
            return
        if elapsed > self.verify_window_s and status.step <= target_step:
            # 重启后没有推进：升级
            record.verified = False
            record.verified_ts = self.clock()
            record.escalated = True
            self.records.append(record)
            self._current = None
            self.phase = PHASE_FAULTED
            self._log("recovery_failed", f"step={status.step} 未超过 {target_step}，升级处理")
            self._escalate(record, status)

    def _escalate(self, record: RecoveryRecord, status: JobStatus) -> None:
        current = S.get(record.strategy)
        nxt = S.ESCALATION.get(current.name, S.STRATEGY_FAIL_FAST)
        self._log("escalate", f"{current.name} -> {nxt}")
        if nxt == S.STRATEGY_FAIL_FAST:
            self.phase = PHASE_FAILED
            self.stop()
            return
        # 用升级后的策略再执行一次（保持同一故障语义）
        plan = S.Plan(
            strategy=nxt,
            fault_code=record.fault_code,
            fault_name=record.fault_name,
            severity=record.severity,
            target=record.target,
            reason=f"上一策略 {current.name} 验证失败，升级为 {nxt}",
            actions=list(S.get(nxt).actions),
            risk=S.get(nxt).risk,
            expected_downtime_s=S.get(nxt).downtime_s,
        )
        self._current_plan = plan
        self._current_fault = None
        self._failure_step = record.failure_step

    # ---------- 杂项 ----------

    def _last_ckpt_step(self) -> int:
        latest = self.store.latest()
        return latest.step if latest else 0

    def request_stop(self) -> None:
        self._stop_requested = True


def _write_atomic(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)
