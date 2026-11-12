"""推理任务跨机迁移控制器（支持 vLLM 与多次迁移/回迁）。

一条完整的处置链：

```text
发现异常（心跳/健康检查/退出码/错误日志）
  -> 翻译成统一故障码（UFC）并上报（alert JSON + 控制台）
  -> 保存进度（尽力优雅保存；进程已死则用最后一个批次的进度）
  -> 把状态文件从源节点搬到目标节点（跨机传输是"迁移"的实质）
  -> 在目标节点 --resume 续跑
  -> 验证（进度推进、健康检查、无同类错误、结果不重不漏、效果不回落）
  -> 可选：二次故障 -> 回迁原节点 -> 再验证
```

控制器本身可以跑在任意一台有 SSH 通道的机器上，不需要部署在被管节点里。
"""

from __future__ import annotations

import argparse
import json
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from aiops.faultcode.models import Fault, Target, Vendor, VendorEvent
from aiops.faultcode.normalizer import Normalizer
from aiops.faultcode.taxonomy import load_catalog
from aiops.remote.ssh import HostSpec, RemoteHost, wait_until

PYTHON = "/root/miniconda3/bin/python"


@dataclass
class Timeline:
    rows: list[dict] = field(default_factory=list)

    def add(self, kind: str, detail: str = "", **extra) -> None:
        row = {"ts": round(time.time(), 3), "kind": kind, "detail": detail, **extra}
        self.rows.append(row)
        print(f"[{row['ts']}] {kind}: {detail}", flush=True)


@dataclass
class FaultSpec:
    mode: str = "none"
    at: int = 0

    @classmethod
    def from_dict(cls, data: dict | None) -> "FaultSpec":
        data = data or {}
        return cls(mode=str(data.get("mode", "none")), at=int(data.get("at", 0)))


@dataclass
class JobSpec:
    name: str
    total: int
    batch: int
    ckpt_every: int
    port: int
    qps: float = 0.0
    backend: str = "infer"           # infer | vllm
    model: str = ""
    vllm_port: int = 18200
    remote_root: str = "/root/aiops"
    grace_s: float = 8.0
    hang_s: float = 6.0
    verify_samples: int = 24

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        job = data.get("job", {}) or {}
        run = data.get("run", {}) or {}
        return cls(
            name=str(job.get("name", "infer-demo")),
            total=int(job.get("total", 1200)),
            batch=int(job.get("batch", 8)),
            ckpt_every=int(job.get("ckpt_every", 64)),
            port=int(job.get("port", 18000)),
            qps=float(job.get("qps", 0.0)),
            backend=str(job.get("backend", "infer")),
            model=str(job.get("model", "")),
            vllm_port=int(job.get("vllm_port", 18200)),
            remote_root=str(job.get("remote_root", "/root/aiops")),
            grace_s=float(run.get("grace_s", 8.0)),
            hang_s=float(run.get("hang_s", 6.0)),
            verify_samples=int(run.get("verify_samples", 24)),
        )


class InferenceMigration:
    def __init__(self, config: dict, out_dir: Path, keep_running: bool = False):
        self.config = config
        hosts = config.get("hosts", {}) or {}
        self.source_spec = HostSpec.from_dict("source", hosts["source"])
        self.target_spec = HostSpec.from_dict("target", hosts["target"])
        self.job = JobSpec.from_dict(config)
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.keep_running = keep_running
        self.timeline = Timeline()
        self.catalog = load_catalog()
        self.normalizer = Normalizer(self.catalog)
        self.alert: dict | None = None
        self.results: list[dict] = []

        run = config.get("run", {}) or {}
        self.failback = bool(run.get("failback", False))
        self.fault_spec = FaultSpec.from_dict(config.get("fault", {}))
        self.failback_fault = FaultSpec.from_dict(run.get("failback_fault", {}))

    # ---------- 路径 / 命令 ----------

    def run_dir(self, host: HostSpec) -> str:
        return f"{self.job.remote_root}/runs/{self.job.name}-{host.name}"

    def worker_module(self) -> str:
        return "demo.inference.vllm_worker" if self.job.backend == "vllm" else "demo.inference.worker"

    def vllm_port(self, host: HostSpec) -> int:
        return int(host.vllm_port or self.job.vllm_port)

    def port(self, host: HostSpec) -> int:
        return int(host.worker_port or self.job.port)

    def start_cmd(self, host: HostSpec, resume: bool, fault: FaultSpec | None) -> str:
        run_dir = self.run_dir(host)
        args = [
            PYTHON, "-m", self.worker_module(),
            "--job", self.job.name,
            "--total", str(self.job.total),
            "--port", str(self.port(host)),
            "--node", host.display,
            "--state-dir", f"{run_dir}/state",
            "--run-dir", run_dir,
            "--control-dir", f"{run_dir}/ctl",
        ]
        if self.job.backend == "vllm":
            args += [
                "--model", self.job.model,
                "--vllm-port", str(self.vllm_port(host)),
                "--qps", str(self.job.qps),
            ]
        else:
            args += [
                "--batch", str(self.job.batch),
                "--ckpt-every", str(self.job.ckpt_every),
                "--qps", str(self.job.qps),
            ]
        if resume:
            args.append("--resume")
        if fault and fault.mode not in ("", "none") and fault.at > 0:
            args += ["--fault", fault.mode, "--fault-at", str(fault.at)]
        inner = " ".join(shlex.quote(a) for a in args)
        inner = "PATH=/root/miniconda3/bin:$PATH " + inner
        if host.gpu is not None:
            inner = f"CUDA_VISIBLE_DEVICES={host.gpu} " + inner
        return (
            f"cd {shlex.quote(self.job.remote_root)} && mkdir -p {shlex.quote(run_dir)}/ctl "
            f"&& rm -f {shlex.quote(run_dir)}/ctl/* {shlex.quote(run_dir)}/exit_code "
            f"&& setsid nohup bash -c {shlex.quote(inner + f' ; echo $? > {run_dir}/exit_code')} "
            f"> {shlex.quote(run_dir)}/worker.log 2>&1 < /dev/null & echo $!"
        )

    def start_worker(self, host: RemoteHost, resume: bool, fault: FaultSpec | None = None) -> int:
        run_dir = self.run_dir(host.spec)
        host.run(f"pkill -f {shlex.quote('--run-dir ' + run_dir)} ; true")
        out = host.run_detached(self.start_cmd(host.spec, resume, fault), read_timeout=4.0)
        pid = 0
        for line in reversed(out.strip().splitlines()):
            if line.strip().isdigit():
                pid = int(line.strip())
                break
        if pid <= 0:
            rc, pgrep, _ = host.run(f"pgrep -f {shlex.quote('--run-dir ' + run_dir)} | tail -1")
            pid = int(pgrep.strip() or 0) if rc == 0 else 0
        if pid <= 0:
            raise RuntimeError("启动失败：拿不到 worker pid")
        self.timeline.add(
            "worker_start",
            f"{host.display} pid={pid} resume={resume} backend={self.job.backend}",
            pid=pid,
        )
        return pid

    # ---------- 读取 ----------

    def heartbeat(self, host: RemoteHost) -> dict | None:
        run_dir = self.run_dir(host.spec)
        for attempt in ("1", "2", "3", "4"):
            path = f"{run_dir}/hb/attempt-{attempt}/rank-0.json"
            if host.exists(path):
                try:
                    return json.loads(host.read_text(path))
                except json.JSONDecodeError:
                    continue
        return None

    def exit_code(self, host: RemoteHost) -> int | None:
        path = f"{self.run_dir(host.spec)}/exit_code"
        if not host.exists(path):
            return None
        try:
            return int(host.read_text(path).strip() or 0)
        except ValueError:
            return None

    def progress(self, host: RemoteHost) -> dict | None:
        path = f"{self.run_dir(host.spec)}/state/progress.json"
        if not host.exists(path):
            return None
        try:
            return json.loads(host.read_text(path))
        except json.JSONDecodeError:
            return None

    def checks(self, host: RemoteHost) -> dict:
        path = f"{self.run_dir(host.spec)}/state/checks.json"
        if not host.exists(path):
            return {}
        try:
            return json.loads(host.read_text(path))
        except json.JSONDecodeError:
            return {}

    # ---------- 检测 / 翻译 / 上报 ----------

    def find_error_line(self, host: RemoteHost) -> str:
        log = host.tail(f"{self.run_dir(host.spec)}/worker.log", lines=120)
        for line in reversed(log.splitlines()):
            if "FATAL" in line or "CUDA error" in line or "out of memory" in line or "engine died" in line:
                return line.strip()
        return ""

    def translate(self, line: str, node: str) -> Fault:
        event = VendorEvent(vendor=Vendor.GENERIC, text=line, component=node, ts=time.time())
        results = self.normalizer.normalize_event(event, Target(node=node))
        if results:
            fault = results[0].fault
            fault.evidence = [line[:300]]
            return fault
        code = self.catalog.get("UFC-JOB-001")
        return Fault(
            code=code,
            target=Target(node=node, job=self.job.name),
            detected_at=time.time(),
            onset_at=time.time(),
            evidence=[f"未识别的错误：{line[:300]}"],
        )

    def publish_alert(self, fault: Fault) -> dict:
        code = fault.code
        alert = {
            "ts": time.time(),
            "job": self.job.name,
            "backend": self.job.backend,
            "source_node": fault.target.node,
            "ufc": code.id,
            "name": code.name,
            "severity": code.severity.value,
            "scope": code.scope.value,
            "summary": code.summary,
            "evidence": fault.evidence,
            "runbook": list(code.runbook),
            "recommended_strategy": list(code.actions),
            "verify": list(code.verify),
        }
        (self.out_dir / "alert.json").write_text(
            json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        text = (
            f"[{alert['severity']}] {alert['ufc']} {alert['name']}\n"
            f"节点：{alert['source_node']}   作业：{alert['job']}（{alert['backend']}）\n"
            f"证据：{'; '.join(alert['evidence'])[:300]}\n"
            f"处置：迁移到备节点并续跑；先执行「{code.runbook[0] if code.runbook else '见手册'}」"
        )
        (self.out_dir / "alert.txt").write_text(text, encoding="utf-8")
        print("\n===== 告警 =====\n" + text + "\n===============\n", flush=True)
        self.alert = alert
        return alert

    # ---------- 保存 / 搬运 ----------

    def save_state(self, host: RemoteHost, pid: int, alive: bool) -> dict:
        run_dir = self.run_dir(host.spec)
        if alive:
            host.write_text(f"{run_dir}/ctl/graceful", "1")
            self.timeline.add("save_requested", "已请求优雅保存")
            deadline = time.time() + self.job.grace_s
            while time.time() < deadline:
                if not host.pid_alive(pid):
                    break
                time.sleep(0.3)
            if host.pid_alive(pid):
                self.timeline.add("save_timeout", f"grace={self.job.grace_s}s 未退出，强制清理")
                host.kill_tree(pid)
        progress = self.progress(host) or {"done": 0}
        self.timeline.add("state_saved", f"done={progress.get('done')}", progress=progress)
        return progress

    def transfer_state(self, source: RemoteHost, target: RemoteHost) -> dict:
        src_run = self.run_dir(source.spec)
        dst_run = self.run_dir(target.spec)
        local = self.out_dir / f"state-transfer-{source.spec.name}-to-{target.spec.name}"
        local.mkdir(parents=True, exist_ok=True)
        target.rm(dst_run)
        target.makedirs(f"{dst_run}/state")

        files = ["state/progress.json", "state/results.jsonl", "state/checks.json"]
        ckpt_dir = f"{src_run}/state/ckpt"
        if source.exists(ckpt_dir):
            for name in source.listdir(ckpt_dir):
                files.append(f"state/ckpt/{name}")

        copied: list[str] = []
        for rel in files:
            local_path = local / rel
            if not source.get(f"{src_run}/{rel}", local_path):
                continue
            target.makedirs(f"{dst_run}/{Path(rel).parent.as_posix()}")
            target.put(local_path, f"{dst_run}/{rel}")
            copied.append(rel)
        total_bytes = sum((local / rel).stat().st_size for rel in copied if (local / rel).exists())
        self.timeline.add(
            "state_transferred",
            f"{len(copied)} 个文件 / {total_bytes} 字节: {', '.join(copied)}",
            files=copied,
            bytes=total_bytes,
        )
        return {"files": copied, "bytes": total_bytes}

    # ---------- 验证 ----------

    def verify(self, target: RemoteHost, target_pid: int, resume_done: int) -> dict:
        deadline = time.time() + 120
        last_done = resume_done
        while time.time() < deadline:
            time.sleep(1.0)
            hb = self.heartbeat(target) or {}
            done = int(hb.get("step", 0))
            health = target.http_status(self.port(target.spec))
            if done >= resume_done + self.job.verify_samples and health == 200:
                integrity = self.check_integrity(target, done)
                if not integrity.get("ok"):
                    return {
                        "verified": False,
                        "done": done,
                        "reason": f"完整性校验失败: {integrity}",
                        "integrity": integrity,
                    }
                return {"verified": True, "done": done, "health": health, "integrity": integrity}
            if not target.pid_alive(target_pid):
                return {"verified": False, "done": done, "reason": "target 进程退出"}
            last_done = done
        return {"verified": False, "done": last_done, "reason": "验证超时"}

    def check_integrity(self, host: RemoteHost, done: int) -> dict:
        run_dir = self.run_dir(host.spec)
        path = f"{run_dir}/state/results.jsonl"
        if not host.exists(path):
            return {"ok": False, "reason": "结果文件不存在"}
        progress = self.progress(host) or {}
        committed = int(progress.get("done", done))
        local = self.out_dir / "verify-results.jsonl"
        if not host.get(path, local):
            return {"ok": False, "reason": "下载结果文件失败"}
        indices: list[int] = []
        for line in local.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                indices.append(int(json.loads(line)["i"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                return {"ok": False, "reason": f"结果行无法解析: {line[:80]}"}
        rows = len(indices)
        head = indices[:committed]
        contiguous = head == list(range(committed))
        duplicates = len(head) != len(set(head))
        # 推理服务比训练快得多，校验时 worker 仍在推进；只要求"已提交部分连续且唯一"，
        # 允许文件比进度多出一段（最多 64 条）以消除这种采样竞态。
        in_flight_ok = rows - committed <= max(1, self.job.batch) * 8 + 64
        ok = contiguous and not duplicates and rows >= committed and in_flight_ok
        return {
            "ok": ok,
            "rows": rows,
            "committed": committed,
            "contiguous": contiguous,
            "duplicates": duplicates,
            "in_flight": rows - committed,
            "first": indices[0] if indices else None,
            "last": indices[-1] if indices else None,
        }

    # ---------- 监控 ----------

    def monitor(self, host: RemoteHost, pid: int) -> tuple[Fault | None, int, float]:
        failure_done = 0
        deadline = time.time() + 300
        while time.time() < deadline:
            time.sleep(0.5)
            hb = self.heartbeat(host) or {}
            done = int(hb.get("step", 0))
            failure_done = max(failure_done, done)
            code = self.exit_code(host)
            alive = host.pid_alive(pid)
            age = time.time() - float(hb.get("ts", time.time()))
            health = host.http_status(self.port(host.spec))

            line = ""
            reason = ""
            if code is None and not alive:
                time.sleep(0.4)
                code = self.exit_code(host)
            if code is not None and code not in (0, 42):
                line = self.find_error_line(host)
                reason = f"进程退出 rc={code}"
            elif not alive:
                line = self.find_error_line(host)
                reason = "进程消失"
            elif hb and age > self.job.hang_s:
                reason = f"心跳 {age:.1f}s 未更新"
            elif health == 0 and hb and age > 2.0:
                line = self.find_error_line(host)
                reason = f"健康检查失败且心跳停摆 {age:.1f}s"
            elif health not in (200, 0):
                line = self.find_error_line(host)
                reason = f"健康检查异常 HTTP {health}"

            if reason:
                detect_ts = time.time()
                onset_ts = float(hb.get("ts", 0) or 0) or detect_ts
                self.timeline.add(
                    "fault_detected",
                    f"{host.display}: {reason}（已完成 {failure_done} 样本）；{line[:200]}",
                )
                fault = self.translate(line, host.display)
                fault.detected_at = detect_ts
                fault.onset_at = onset_ts
                fault.detection_latency_s = max(0.0, detect_ts - onset_ts)
                return fault, failure_done, onset_ts
            if code == 0 and done >= self.job.total:
                return None, done, 0.0
        return None, failure_done, 0.0

    # ---------- 主流程 ----------

    def run(self) -> dict:
        started = time.time()
        source = RemoteHost(self.source_spec)
        target = RemoteHost(self.target_spec)
        hops: list[tuple[RemoteHost, RemoteHost, FaultSpec]] = [(source, target, self.fault_spec)]
        if self.failback:
            hops.append((target, source, self.failback_fault))

        try:
            module = self.worker_module()
            source.run_ok(
                f"cd {shlex.quote(self.job.remote_root)} && ls {shlex.quote('demo/inference/' + module.split('.')[-1] + '.py')}"
            )
            source.run("pkill -f demo.inference.worker; pkill -f demo.inference.vllm_worker; sleep 1; true")
            if bool((self.config.get("run") or {}).get("clean_gpu", False)):
                # 上一次实验可能留下持显存的引擎；演示环境里直接清空计算进程
                source.run(
                    "nvidia-smi --query-compute-apps=pid --format=csv,noheader "
                    "| xargs -r kill -9; sleep 2; true"
                )
                self.timeline.add("gpu_cleaned", "已清理两张卡上的残留计算进程")
            # 清理上一轮残留（心跳/状态/控制文件），否则会把旧心跳当成"当前挂起"
            for node in (hops[0][0], hops[-1][1]):
                node.rm(self.run_dir(node.spec))
            self.timeline.add("connect", f"backend={self.job.backend} hops={len(hops)}")

            pid = self.start_worker(hops[0][0], resume=False, fault=hops[0][2])
            if not wait_until(
                lambda: (self.heartbeat(hops[0][0]) or {}).get("step", 0) > 0, timeout_s=300
            ):
                raise RuntimeError("首个节点未产出进度")

            for idx, (cur, nxt, _fault_spec) in enumerate(hops):
                fault, failure_done, _ = self.monitor(cur, pid)
                if fault is None:
                    raise RuntimeError(f"第 {idx} 跳：监控窗口内没有故障（检查注入参数）")
                self.publish_alert(fault)
                self.timeline.add(
                    "translated",
                    f"{fault.code.id} {fault.code.name} [{fault.severity.value}]",
                    ufc=fault.code.id,
                )

                alive = cur.pid_alive(pid)
                progress = self.save_state(cur, pid, alive)
                saved_done = int(progress.get("done", 0))
                rpo = max(0, failure_done - saved_done)

                transferred = self.transfer_state(cur, nxt)
                cur.write_text(
                    f"{self.job.remote_root}/runs/quarantine-{cur.spec.name}.json",
                    json.dumps(
                        {"node": cur.display, "ufc": fault.code.id, "ts": time.time()},
                        ensure_ascii=False,
                    ),
                )
                self.timeline.add("quarantine", f"{cur.display} 已标记隔离")

                next_fault = hops[idx + 1][2] if idx + 1 < len(hops) else None
                pid = self.start_worker(nxt, resume=True, fault=next_fault)
                if not wait_until(
                    lambda: (self.heartbeat(nxt) or {}).get("step", 0) >= saved_done, timeout_s=300
                ):
                    raise RuntimeError(f"第 {idx} 跳：{nxt.display} 未恢复进度")
                resumed_ts = time.time()
                self.timeline.add(
                    "resumed",
                    f"{nxt.display} 从 {saved_done} 继续",
                    done=(self.heartbeat(nxt) or {}).get("step"),
                )

                verify = self.verify(nxt, pid, saved_done)
                verified_ts = time.time()
                history = self.checks(nxt).get("history", [])
                latest = history[-1] if history else {}
                record = {
                    "hop": idx + 1,
                    "direction": f"{cur.display}->{nxt.display}",
                    "ufc": fault.code.id,
                    "ufc_name": fault.code.name,
                    "severity": fault.severity.value,
                    "failure_done": failure_done,
                    "saved_done": saved_done,
                    "rpo_samples": rpo,
                    "mttd_s": round(max(0.0, fault.detection_latency_s), 2),
                    "save_s": round(resumed_ts - fault.detected_at, 2),
                    "mttr_s": round(verified_ts - fault.detected_at, 2),
                    "verified": bool(verify.get("verified")),
                    "verify": verify,
                    "transferred": transferred,
                    "checks": latest,
                }
                self.results.append(record)
                self.timeline.add(
                    "verified" if record["verified"] else "verify_failed",
                    f"hop={idx + 1} done={verify.get('done')} integrity={verify.get('integrity')} "
                    f"probe={latest.get('probe', {}).get('hash')} "
                    f"replay={latest.get('replay', {}).get('matched')}/{latest.get('replay', {}).get('checked')}",
                )
                if not record["verified"]:
                    break

            final_host = hops[-1][1]
            history = self.checks(final_host).get("history", [])
            probes = [c.get("probe", {}).get("hash") for c in history if c.get("probe")]
            calib = [
                {
                    "node": c.get("node"),
                    "accuracy": (c.get("probe", {}).get("calibration") or {}).get("accuracy"),
                }
                for c in history
            ]
            replay = [
                {
                    "node": c.get("node"),
                    "matched": (c.get("replay") or {}).get("matched"),
                    "checked": (c.get("replay") or {}).get("checked"),
                }
                for c in history
            ]
            determinism = {
                "probe_hashes": probes,
                "probe_identical": (len(set(probes)) == 1) if probes else None,
                "calibration": calib,
                "replay": replay,
                "replay_all_matched": all(
                    r["matched"] == r["checked"] for r in replay if r["checked"]
                ),
            }

            if not self.keep_running:
                final_host.write_text(f"{self.run_dir(final_host.spec)}/ctl/force_exit", "1")
                self.timeline.add("target_stopped", "演示结束后停止最终节点 worker")

            summary = {
                "job": self.job.name,
                "backend": self.job.backend,
                "hops": len(self.results),
                "directions": [r["direction"] for r in self.results],
                "ufcs": [r["ufc"] for r in self.results],
                "verified": bool(self.results) and all(r["verified"] for r in self.results),
                "total_rpo_samples": sum(r["rpo_samples"] for r in self.results),
                "total_mttr_s": round(sum(r["mttr_s"] for r in self.results), 2),
                "total_s": round(time.time() - started, 1),
                "determinism": determinism,
                "records": self.results,
                "alert": self.alert,
            }
            self.write_artifacts(summary)
            return summary
        finally:
            source.close()
            target.close()

    def write_artifacts(self, summary: dict) -> None:
        (self.out_dir / "timeline.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in self.timeline.rows) + "\n",
            encoding="utf-8",
        )
        (self.out_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        det = summary["determinism"]
        lines = [
            f"# 推理任务迁移报告：{summary['job']}（{summary['backend']}）",
            "",
            f"- 迁移次数：{summary['hops']}（{' → '.join(summary['directions'])}）",
            f"- 故障码：{'、'.join(summary['ufcs'])}",
            f"- 总丢失样本（RPO）：{summary['total_rpo_samples']}",
            f"- 总恢复时延：{summary['total_mttr_s']}s / 总耗时 {summary['total_s']}s",
            f"- 验证：{'全部通过' if summary['verified'] else '存在失败'}",
            "",
            "## 各跳明细",
            "",
            "| 跳 | 方向 | 故障码 | 故障进度 | 保存点 | RPO | MTTD | MTTR | 验证 | 回放一致 |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in summary["records"]:
            rep = r.get("checks", {}).get("replay") or {}
            lines.append(
                f"| {r['hop']} | {r['direction']} | {r['ufc']} | {r['failure_done']} | {r['saved_done']} "
                f"| {r['rpo_samples']} | {r['mttd_s']}s | {r['mttr_s']}s | {'✅' if r['verified'] else '❌'} "
                f"| {rep.get('matched', '-')}/{rep.get('checked', '-')} |"
            )
        lines += [
            "",
            "## 结果一致性（是否偏移）",
            "",
            f"- 各节点 probe 哈希：{det['probe_hashes']}",
            f"- probe 完全一致：{det['probe_identical']}",
            f"- 校准集准确率：{det['calibration']}",
            f"- 回放比对：{det['replay']}",
            f"- 回放全部一致：{det['replay_all_matched']}",
        ]
        (self.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="推理任务跨机迁移（支持 vLLM 与回迁）")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", default="results/inference-migration")
    parser.add_argument("--keep-running", action="store_true")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    controller = InferenceMigration(config, out, keep_running=args.keep_running)
    result = controller.run()
    print("\n===== 结果 =====")
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ("records", "alert")},
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if result.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
