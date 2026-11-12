"""故障场景回放器：把 YAML 剧本变成可复现的实验。

一个场景 = 一个作业 + 若干"在某个步数注入的故障" + 期望的策略与 RPO。
注入分两类：

* **控制文件**：``die`` / ``hang`` / ``throttle`` / ``nan``，直接改变仿真训练器行为；
* **厂商事件**：Xid / amdgpu / slog 的一行日志，走生产同一条翻译链。

跑完会落三份产物：``timeline.jsonl``（全过程）、``result.json``（结构化结果）、
``summary.md``（人读）。多个场景聚合成 ``results/report.md``。
"""

from __future__ import annotations

import json
import shutil
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from aiops import metrics as metrics_mod
from aiops.collect.base import Snapshot
from aiops.detect.rules import Detector
from aiops.detect.straggler import BenchMonitor, BenchResult
from aiops.detect.telemetry import TelemetryMonitor
from aiops.faultcode.models import Vendor, VendorEvent
from aiops.faultcode.taxonomy import Catalog, load_catalog
from aiops.policy.engine import PolicyEngine
from aiops.supervisor.checkpoint import CheckpointStore
from aiops.supervisor.supervisor import JobSpec, JobStatus, Probe, RunResult, Supervisor
from demo.probe import HeartbeatProbe

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------- 场景定义


@dataclass
class Injection:
    at_step: int = 0
    at_s: float = 0.0
    action: str = "event"       # die_rank / hang_rank / throttle_rank / nan / event
    rank: int = 1
    value: float = 1.0
    vendor: str = "nvidia"
    code: str = ""
    text: str = ""
    applied: bool = False


@dataclass
class Scenario:
    name: str
    description: str = ""
    steps: int = 200
    backend: str = "sim"
    platform: str = "any"
    world_size: int = 2
    step_sleep: float = 0.08
    ckpt_every: int = 20
    node: str = "gpu-node-15"
    spare_nodes: list[str] = field(default_factory=list)
    grace_s: float = 5.0
    injections: list[Injection] = field(default_factory=list)
    bench: dict = field(default_factory=dict)
    threshold_overrides: list[dict] = field(default_factory=list)
    expect: dict = field(default_factory=dict)


def load_scenario(path: str | Path) -> Scenario:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    job = data.get("job", {}) or {}
    injections = [
        Injection(
            at_step=int(item.get("at_step", 0)),
            at_s=float(item.get("at_s", 0.0)),
            action=str(item.get("action", "event")),
            rank=int(item.get("rank", 1)),
            value=float(item.get("value", 1.0)),
            vendor=str(item.get("vendor", "nvidia")),
            code=str(item.get("code", "")),
            text=str(item.get("text", "")),
        )
        for item in data.get("injections", []) or []
    ]
    return Scenario(
        name=str(data.get("name") or Path(path).stem),
        description=str(data.get("description", "")),
        steps=int(job.get("steps", 200)),
        backend=str(job.get("backend", "sim")),
        platform=str(job.get("platform", "any")),
        world_size=int(job.get("world_size", 2)),
        step_sleep=float(job.get("step_sleep", 0.08)),
        ckpt_every=int(job.get("ckpt_every", 20)),
        node=str(job.get("node", "gpu-node-15")),
        spare_nodes=[str(n) for n in job.get("spare_nodes", []) or []],
        grace_s=float(job.get("grace_s", 5.0)),
        injections=injections,
        bench=dict(data.get("bench", {}) or {}),
        threshold_overrides=list(data.get("threshold_overrides", []) or []),
        expect=dict(data.get("expect", {}) or {}),
    )


def apply_threshold_overrides(catalog: Catalog, overrides: Iterable[Mapping[str, Any]]) -> Catalog:
    """演练模式下调低阈值（例如把挂起判定从 300s 调到 8s）。

    生产环境的阈值不应为了让 demo 好看而改；这里只接受显式的场景级覆盖，
    并且会在报告里标出被覆盖的阈值。
    """

    codes = list(catalog)
    for override in overrides:
        code_id = override["code"]
        metric = override["metric"]
        value = float(override["value"])
        for idx, code in enumerate(codes):
            if code.id != code_id:
                continue
            new_signals = tuple(
                replace(rule, value=value) if rule.metric == metric else rule
                for rule in code.signals
            )
            if new_signals == code.signals:
                raise KeyError(f"{code_id} 没有指标 {metric}，无法覆盖阈值")
            codes[idx] = replace(code, signals=new_signals)
    return Catalog(codes, catalog.meta)


# ---------------------------------------------------------------- 注入器


class Injector:
    def __init__(self, scenario: Scenario, control_dir: Path, clock=time.time):
        self.scenario = scenario
        self.control_dir = control_dir
        self.clock = clock
        self.start = clock()
        self.pending_events: list[VendorEvent] = []
        self.applied: list[dict] = []

    def _due(self, injection: Injection, status: JobStatus) -> bool:
        if injection.applied:
            return False
        if injection.at_step and status.step >= injection.at_step:
            return True
        if injection.at_s and (self.clock() - self.start) >= injection.at_s:
            return True
        return False

    def tick(self, status: JobStatus) -> None:
        for injection in self.scenario.injections:
            if not self._due(injection, status):
                continue
            injection.applied = True
            self._apply(injection, status)

    def _apply(self, injection: Injection, status: JobStatus) -> None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": round(self.clock(), 3),
            "step": status.step,
            "action": injection.action,
            "rank": injection.rank,
            "value": injection.value,
        }
        if injection.action == "die_rank":
            (self.control_dir / f"die.{injection.rank}").write_text("1", encoding="utf-8")
        elif injection.action == "hang_rank":
            (self.control_dir / f"hang.{injection.rank}").write_text("1", encoding="utf-8")
        elif injection.action == "throttle_rank":
            (self.control_dir / f"throttle.{injection.rank}").write_text(
                str(injection.value), encoding="utf-8"
            )
        elif injection.action == "nan":
            (self.control_dir / "nan").write_text("1", encoding="utf-8")
        elif injection.action == "event":
            self.pending_events.append(
                VendorEvent(
                    vendor=Vendor.parse(injection.vendor),
                    code=injection.code,
                    text=injection.text or f"{injection.vendor} fault {injection.code}",
                    component=status.node or self.scenario.node,
                    ts=self.clock(),
                )
            )
        else:
            raise ValueError(f"未知注入动作: {injection.action}")
        self.applied.append({**record, "label": injection.action})

    def drain(self) -> list[VendorEvent]:
        events, self.pending_events = self.pending_events, []
        return events


# ---------------------------------------------------------------- 运行器


@dataclass
class ScenarioOutcome:
    scenario: Scenario
    result: RunResult
    out_dir: Path
    expect_ok: bool = True
    expect_notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "scenario": self.scenario.name,
            "description": self.scenario.description,
            "expect": self.scenario.expect,
            "expect_ok": self.expect_ok,
            "expect_notes": self.expect_notes,
            **self.result.summary(),
        }


def run_scenario(
    scenario: Scenario | str | Path,
    out_root: str | Path = "results",
    catalog: Catalog | None = None,
    max_seconds: float | None = None,
) -> ScenarioOutcome:
    if not isinstance(scenario, Scenario):
        scenario = load_scenario(scenario)
    catalog = catalog or load_catalog()
    if scenario.threshold_overrides:
        catalog = apply_threshold_overrides(catalog, scenario.threshold_overrides)

    if scenario.platform == "linux" and sys.platform != "linux":
        raise RuntimeError(
            f"场景 {scenario.name} 需要 Linux（当前 {sys.platform}）："
            "torchrun 依赖 TCPStore，本机 torch 构建未启用 libuv"
        )
    out_dir = Path(out_root) / scenario.name
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    run_dir = out_dir / "run"
    ckpt_dir = out_dir / "ckpt"
    ctl_dir = run_dir / "ctl"
    for path in (run_dir, ckpt_dir, ctl_dir):
        path.mkdir(parents=True, exist_ok=True)

    store = CheckpointStore(ckpt_dir, scenario.name)
    detector = Detector(catalog)
    policy = PolicyEngine(catalog)
    telemetry = TelemetryMonitor(sources=["heartbeat"], expected_interval_s=0.5, stale_factor=6)
    bench = None
    if scenario.bench:
        metric = str(scenario.bench.get("metric", "allreduce_gbps"))
        base = float(scenario.bench.get("baseline", 0))
        bench = BenchMonitor(fleet_baseline={metric: base} if base else {})
        for node, value in (scenario.bench.get("values") or {}).items():
            bench.record(BenchResult(node=str(node), gpu="0", metric=metric, value=float(value)))
    probe_inner = HeartbeatProbe(run_dir, store, scenario.name, scenario.node, scenario.world_size)
    injector = Injector(scenario, ctl_dir)

    def probe() -> Probe:
        p = probe_inner()
        injector.tick(p.status)
        events = injector.drain()
        if events:
            p.snapshot.events.extend(events)
        return p

    common = [
        "--job",
        scenario.name,
        "--steps",
        str(scenario.steps),
        "--ckpt-dir",
        str(ckpt_dir),
        "--run-dir",
        str(run_dir),
        "--control-dir",
        str(ctl_dir),
        "--ckpt-every",
        str(scenario.ckpt_every),
        "--node",
        scenario.node,
    ]
    if scenario.backend == "torch":
        argv = [
            sys.executable,
            "-m",
            "demo.torch_launcher",
            "--world-size",
            str(scenario.world_size),
            "--run-dir",
            str(run_dir),
            *common,
        ]
        world_size_flag = "--world-size"
    else:
        argv = [
            sys.executable,
            "-m",
            "demo.sim_trainer",
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(ckpt_dir),
            "--control-dir",
            str(ctl_dir),
            "--job",
            scenario.name,
            "--steps",
            str(scenario.steps),
            "--world-size",
            str(scenario.world_size),
            "--step-sleep",
            str(scenario.step_sleep),
            "--ckpt-every",
            str(scenario.ckpt_every),
            "--node",
            scenario.node,
        ]
        world_size_flag = "--world-size"

    spec = JobSpec(
        name=scenario.name,
        argv=argv,
        kind="train",
        node=scenario.node,
        world_size=scenario.world_size,
        ckpt_dir=str(ckpt_dir),
        control_dir=str(ctl_dir),
        stop_mode="control",
        grace_s=scenario.grace_s,
        spare_nodes=list(scenario.spare_nodes),
        world_size_flag=world_size_flag,
        cwd=str(REPO_ROOT),
    )

    supervisor = Supervisor(
        spec=spec,
        store=store,
        detector=detector,
        policy=policy,
        probe=probe,
        telemetry=telemetry,
        poll_s=0.05,
        verify_steps=3,
        verify_window_s=12.0,
        bench=bench,
    )
    budget = max_seconds or (scenario.steps * scenario.step_sleep * 6 + 120)
    result = supervisor.run(max_seconds=budget)

    _write_artifacts(out_dir, scenario, result, injector, catalog, supervisor)
    outcome = ScenarioOutcome(scenario=scenario, result=result, out_dir=out_dir)
    _check_expectations(outcome)
    return outcome


def _check_expectations(outcome: ScenarioOutcome) -> None:
    expect = outcome.scenario.expect or {}
    notes: list[str] = []
    ok = True
    strategies = [r.strategy for r in outcome.result.recoveries]
    strategies += [
        str(e.get("strategy")) for e in outcome.result.timeline if e.get("kind") == "decision"
    ]
    if "any_strategy" in expect:
        allowed = set(expect["any_strategy"])
        if not any(s in allowed for s in strategies):
            ok = False
            notes.append(f"策略 {strategies} 不在期望集合 {sorted(allowed)} 内")
    if "max_rpo_steps" in expect and outcome.result.recoveries:
        worst = max(r.rpo_steps for r in outcome.result.recoveries)
        if worst > expect["max_rpo_steps"]:
            ok = False
            notes.append(f"RPO {worst} 步超过预算 {expect['max_rpo_steps']}")
    if expect.get("recovered", True) and outcome.result.recoveries:
        if not any(r.verified for r in outcome.result.recoveries):
            ok = False
            notes.append("没有一次恢复通过验证")
    if expect.get("min_recoveries") and len(outcome.result.recoveries) < expect["min_recoveries"]:
        ok = False
        notes.append(f"恢复次数 {len(outcome.result.recoveries)} 少于期望 {expect['min_recoveries']}")
    outcome.expect_ok = ok
    outcome.expect_notes = notes


def _write_artifacts(
    out_dir: Path,
    scenario: Scenario,
    result: RunResult,
    injector: Injector,
    catalog: Catalog,
    supervisor: Supervisor | None = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "timeline.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in result.timeline) + "\n",
        encoding="utf-8",
    )
    (out_dir / "injections.json").write_text(
        json.dumps(injector.applied, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "scenario.yaml").write_text(
        yaml.safe_dump(
            {
                "name": scenario.name,
                "description": scenario.description,
                "job": {
                    "steps": scenario.steps,
                    "world_size": scenario.world_size,
                    "step_sleep": scenario.step_sleep,
                    "ckpt_every": scenario.ckpt_every,
                    "node": scenario.node,
                    "spare_nodes": scenario.spare_nodes,
                },
                "expect": scenario.expect,
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    payload = {
        "summary": result.summary(),
        "recoveries": [r.as_dict() for r in result.recoveries],
        "faults": result.faults,
        "injections": injector.applied,
        "catalog_version": catalog.meta.get("version"),
    }
    (out_dir / "result.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "summary.md").write_text(_summary_md(outcome_scenario=scenario, result=result), encoding="utf-8")
    metrics_mod.write(
        out_dir / "metrics.prom",
        result.faults,
        [r.as_dict() for r in result.recoveries],
        [f.as_dict() for f in (supervisor.detector.active.values() if supervisor else [])],
    )


def _summary_md(outcome_scenario: Scenario, result: RunResult) -> str:
    summary = result.summary()
    lines = [
        f"# 场景：{outcome_scenario.name}",
        "",
        outcome_scenario.description,
        "",
        "| 指标 | 值 |",
        "| --- | --- |",
        f"| 是否跑完 | {summary['ok']} |",
        f"| 完成步数 | {summary['steps_completed']} |",
        f"| 检测故障数 | {summary['faults']} |",
        f"| 恢复次数 / 成功 | {summary['recoveries']} / {summary['recovered']} |",
        f"| 平均 MTTD | {summary['mttd_avg_s']} s |",
        f"| 平均 MTTR | {summary['mttr_avg_s']} s |",
        f"| 平均停机 | {summary['downtime_avg_s']} s |",
        f"| 平均 RPO | {summary['rpo_steps_avg']} 步 |",
        f"| 最大 RPO | {summary['rpo_steps_max']} 步 |",
        f"| 仅周期检查点会丢的步数 | {summary['rpo_without_emergency_avg']} 步 |",
        f"| 升级次数 | {summary['escalations']} |",
        f"| 墙钟时间 | {summary['wall_s']} s |",
        "",
        "## 恢复明细",
        "",
        "| 故障 | 策略 | 检查点步 | 失败步 | RPO | 停机(s) | MTTR(s) | 验证 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in result.recoveries:
        lines.append(
            f"| {r.fault_code} {r.fault_name} | {r.strategy} | {r.ckpt_step} | {r.failure_step} | "
            f"{r.rpo_steps} | {r.downtime_s:.1f} | {r.mttr_s:.1f} | {'✅' if r.verified else '❌'} |"
        )
    if result.faults:
        lines += ["", "## 检测到的故障", "", "| 码 | 名称 | 等级 | 对象 | 证据 |", "| --- | --- | --- | --- | --- |"]
        for f in result.faults:
            evidence = "; ".join(str(e) for e in f.get("evidence", [])[:2])
            lines.append(
                f"| {f['code']} | {f['name']} | {f['severity']} | {f['target']} | {evidence} |"
            )
    return "\n".join(lines) + "\n"


def run_all(paths: Iterable[str | Path], out_root: str | Path = "results") -> list[ScenarioOutcome]:
    catalog = load_catalog()
    outcomes: list[ScenarioOutcome] = []
    for path in paths:
        started = time.time()
        print(f"[run] {Path(path).stem} ...", flush=True)
        outcome = run_scenario(path, out_root=out_root, catalog=catalog)
        outcomes.append(outcome)
        s = outcome.result.summary()
        print(
            f"      ok={s['ok']} faults={s['faults']} recoveries={s['recoveries']} "
            f"recovered={s['recovered']} rpo_max={s['rpo_steps_max']} "
            f"wall={time.time() - started:.1f}s expect_ok={outcome.expect_ok}",
            flush=True,
        )
    write_report(outcomes, Path(out_root) / "report.md")
    return outcomes


def write_report(outcomes: Iterable[ScenarioOutcome], path: Path = Path("results/report.md")) -> Path:
    outcomes = list(outcomes)
    rows = [o.as_dict() for o in outcomes]
    lines = [
        "# 自动运维场景实验报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "| 场景 | 故障数 | 恢复 | 成功 | 平均停机(s) | 平均 MTTD(s) | 平均 MTTR(s) | 平均 RPO(步) | 仅周期检查点 RPO | 期望达标 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['faults']} | {row['recoveries']} | {row['recovered']} | "
            f"{row['downtime_avg_s']} | {row['mttd_avg_s']} | {row['mttr_avg_s']} | "
            f"{row['rpo_steps_avg']} | {row['rpo_without_emergency_avg']} | "
            f"{'✅' if row['expect_ok'] else '❌ ' + '; '.join(row['expect_notes'])} |"
        )
    total_recoveries = sum(r["recoveries"] for r in rows)
    total_recovered = sum(r["recovered"] for r in rows)
    lines += [
        "",
        f"合计：{len(rows)} 个场景，{total_recoveries} 次恢复，{total_recovered} 次通过验证。",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
