"""推理任务跨机迁移控制器。

一条完整的处置链：

```text
发现异常（心跳/健康检查/退出码/错误日志）
  -> 翻译成统一故障码（UFC）并上报（alert JSON + 控制台）
  -> 保存进度（尽力优雅保存；进程已死则用最后一个批次的进度）
  -> 把状态文件从源机搬到目标机（跨机传输是"迁移"的实质）
  -> 在目标机 --resume 续跑
  -> 验证（进度推进、健康检查通过、无同类错误、吞吐达标）
  -> 落盘时间线与结果（MTTD / MTTR / 停机 / RPO / 完整性）
```

控制器本身可以跑在任意一台有 SSH 通道的机器上（这里是我的本机），
不需要部署在被管节点里——这也是跨机迁移的前提。
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
class JobSpec:
    name: str
    total: int
    batch: int
    ckpt_every: int
    port: int
    qps: float = 0.0
    remote_root: str = "/root/aiops"
    fault: str = "device-assert"
    fault_at: int = 0
    grace_s: float = 8.0
    hang_s: float = 6.0
    verify_samples: int = 24

    @classmethod
    def from_dict(cls, data: dict) -> "JobSpec":
        job = data.get("job", {}) or {}
        fault = data.get("fault", {}) or {}
        run = data.get("run", {}) or {}
        return cls(
            name=str(job.get("name", "infer-demo")),
            total=int(job.get("total", 2000)),
            batch=int(job.get("batch", 8)),
            ckpt_every=int(job.get("ckpt_every", 128)),
            port=int(job.get("port", 18000)),
            qps=float(job.get("qps", 0.0)),
            remote_root=str(job.get("remote_root", "/root/aiops")),
            fault=str(fault.get("mode", "device-assert")),
            fault_at=int(fault.get("at", 400)),
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
        self.keep_running = keep_running
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.timeline = Timeline()
        self.catalog = load_catalog()
        self.normalizer = Normalizer(self.catalog)
        self.alert: dict | None = None
        self.result: dict = {}

    # ---------- 路径 ----------

    def run_dir(self, host: HostSpec) -> str:
        return f"{self.job.remote_root}/runs/{self.job.name}-{host.name}"

    def start_cmd(self, host: HostSpec, resume: bool) -> str:  # host: HostSpec
        run_dir = self.run_dir(host)
        args = [
            PYTHON, "-m", "demo.inference.worker",
            "--job", self.job.name,
            "--total", str(self.job.total),
            "--batch", str(self.job.batch),
            "--ckpt-every", str(self.job.ckpt_every),
            "--port", str(self.port(host)),
            "--qps", str(self.job.qps),
            "--node", host.display,
            "--state-dir", f"{run_dir}/state",
            "--run-dir", run_dir,
            "--control-dir", f"{run_dir}/ctl",
        ]
        if resume:
            args.append("--resume")
        if not resume and self.job.fault in ("device-assert", "cuda-oom"):
            args += ["--fault", self.job.fault, "--fault-at", str(self.job.fault_at)]
        inner = " ".join(shlex.quote(a) for a in args)
        if host.gpu is not None:
            inner = f"CUDA_VISIBLE_DEVICES={host.gpu} " + inner
        return (
            f"cd {shlex.quote(self.job.remote_root)} && mkdir -p {shlex.quote(run_dir)}/ctl "
            f"&& rm -f {shlex.quote(run_dir)}/exit_code "
            f"&& setsid nohup bash -c {shlex.quote(inner + f' ; echo $? > {run_dir}/exit_code')} "
            f"> {shlex.quote(run_dir)}/worker.log 2>&1 < /dev/null & echo $!"
        )

    # ---------- 启动 / 读取 ----------

    def port(self, host: HostSpec) -> int:
        return int(host.worker_port or self.job.port)

    def start_worker(self, host: RemoteHost, resume: bool) -> int:
        run_dir = self.run_dir(host.spec)
        host.run(f"pkill -f {shlex.quote('--run-dir ' + run_dir)} ; true")
        cmd = self.start_cmd(host.spec, resume)
        out = host.run_detached(cmd, read_timeout=4.0)
        pid = 0
        for line in reversed(out.strip().splitlines()):
            if line.strip().isdigit():
                pid = int(line.strip())
                break
        if pid <= 0:
            rc, pgrep, _ = host.run(
                f"pgrep -f {shlex.quote('--run-dir ' + run_dir)} | tail -1"
            )
            pid = int(pgrep.strip() or 0) if rc == 0 else 0
        if pid <= 0:
            raise RuntimeError("启动失败：拿不到 worker pid")
        self.timeline.add("worker_start", f"{host.display} pid={pid} resume={resume}", pid=pid)
        return pid

    def heartbeat(self, host: RemoteHost) -> dict | None:
        run_dir = self.run_dir(host.spec)
        for attempt in ("1", "2", "3"):
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

    # ---------- 故障判定 ----------

    def find_error_line(self, host: RemoteHost) -> str:
        log = host.tail(f"{self.run_dir(host.spec)}/worker.log", lines=80)
        for line in reversed(log.splitlines()):
            if "FATAL" in line or "CUDA error" in line or "out of memory" in line:
                return line.strip()
        return ""

    def translate(self, line: str, node: str) -> Fault:
        event = VendorEvent(vendor=Vendor.GENERIC, text=line, component=node, ts=time.time())
        results = self.normalizer.normalize_event(event, Target(node=node))
        if results:
            fault = results[0].fault
            fault.evidence = [line[:300]]
            return fault
        # 兜底：能确认作业失败，但厂商码未识别
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
            "source_node": fault.target.node,
            "ufc": code.id,
            "name": code.name,
            "severity": code.severity.value,
            "scope": code.scope.value,
            "summary": code.summary,
            "evidence": fault.evidence,
            "runbook": list(code.runbook),
            "recommended_strategy": [a for a in code.actions],
            "verify": list(code.verify),
        }
        path = self.out_dir / "alert.json"
        path.write_text(json.dumps(alert, ensure_ascii=False, indent=2), encoding="utf-8")
        text = (
            f"[{alert['severity']}] {alert['ufc']} {alert['name']}\n"
            f"节点：{alert['source_node']}   作业：{alert['job']}\n"
            f"证据：{'; '.join(alert['evidence'])[:300]}\n"
            f"处置：迁移到备机并续跑；先执行「{code.runbook[0] if code.runbook else '见手册'}」"
        )
        (self.out_dir / "alert.txt").write_text(text, encoding="utf-8")
        print("\n===== 告警 =====\n" + text + "\n===============\n", flush=True)
        self.alert = alert
        return alert

    # ---------- 保存 / 迁移 ----------

    def save_state(self, host: RemoteHost, pid: int, alive: bool) -> dict:
        """尽力保存：活着就优雅停，死了就用已提交的进度。"""

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
        """源机状态 -> 本机中转 -> 目标机（跨机搬运）。"""

        src_run = self.run_dir(source.spec)
        dst_run = self.run_dir(target.spec)
        local = self.out_dir / "state-transfer"
        local.mkdir(parents=True, exist_ok=True)
        target.rm(dst_run)
        target.makedirs(f"{dst_run}/state")

        files = ["state/progress.json", "state/results.jsonl"]
        # 检查点目录里的标记文件也一起搬，模拟真实迁移要带走完整状态
        for name in source.listdir(f"{src_run}/state/ckpt") if source.exists(f"{src_run}/state/ckpt") else []:
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

    def verify(self, target: RemoteHost, target_pid: int, resume_done: int, source_rate: float) -> dict:
        deadline = time.time() + 60
        baseline_done = resume_done
        last_done = resume_done
        last_ts = time.time()
        while time.time() < deadline:
            time.sleep(1.0)
            hb = self.heartbeat(target) or {}
            done = int(hb.get("step", 0))
            if done > last_done:
                last_ts = time.time()
                last_done = done
            health = target.http_status(self.port(target.spec))
            if done >= baseline_done + self.job.verify_samples and health == 200:
                elapsed = max(1e-6, last_ts - (deadline - 60))
                target_rate = (done - baseline_done) / max(1.0, time.time() - (deadline - 60))
                # 完整性：结果行数必须等于 done，且下标连续
                integrity = self.check_integrity(target, done)
                if not integrity.get("ok"):
                    return {
                        "verified": False,
                        "done": done,
                        "health": health,
                        "reason": f"完整性校验失败: {integrity}",
                        "integrity": integrity,
                    }
                return {
                    "verified": True,
                    "done": done,
                    "health": health,
                    "target_rate": round(target_rate, 2),
                    "source_rate": round(source_rate, 2),
                    "integrity": integrity,
                }
            if not target.pid_alive(target_pid):
                return {"verified": False, "done": done, "reason": "target 进程退出"}
        return {"verified": False, "done": last_done, "reason": "验证超时"}

    def check_integrity(self, host: RemoteHost, done: int) -> dict:
        """校验"结果不重不漏"。

        采样时刻文件可能正好多出正在写入的一批（append 与 commit 之间有窗口），
        因此以**已提交的 progress.done** 为准：前 done 条必须是 0..done-1 且唯一，
        总行数不超过 done + batch。
        """

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
        in_flight_ok = rows - committed <= max(1, self.job.batch)
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

    # ---------- 主流程 ----------

    def run(self) -> dict:
        started = time.time()
        source = RemoteHost(self.source_spec)
        target = RemoteHost(self.target_spec)
        try:
            self.timeline.add("connect", f"source={source.display} target={target.display}")
            source.run_ok(f"cd {shlex.quote(self.job.remote_root)} && ls demo/inference/worker.py")
            source.run("pkill -f demo.inference.worker; sleep 1; true")
            target.run_ok(f"cd {shlex.quote(self.job.remote_root)} && ls demo/inference/worker.py")
            source.rm(self.run_dir(self.source_spec))
            source.makedirs(f"{self.run_dir(self.source_spec)}/ctl")

            pid = self.start_worker(source, resume=False)
            started_ok = wait_until(
                lambda: (self.heartbeat(source) or {}).get("step", 0) > 0, timeout_s=120
            )
            if not started_ok:
                raise RuntimeError("源机 worker 未产出进度")
            baseline_start = time.time()
            baseline_done = int((self.heartbeat(source) or {}).get("step", 0))
            time.sleep(6)
            hb = self.heartbeat(source) or {}
            source_rate = (int(hb.get("step", 0)) - baseline_done) / max(
                1e-6, time.time() - baseline_start
            )
            self.timeline.add("baseline", f"源机吞吐 {source_rate:.1f} samples/s", rate=round(source_rate, 2))

            # ---- 监控：等故障 ----
            fault: Fault | None = None
            failure_done = 0
            detect_ts = 0.0
            onset_ts = 0.0
            poll_deadline = time.time() + 180
            while time.time() < poll_deadline:
                time.sleep(0.5)
                hb = self.heartbeat(source) or {}
                done = int(hb.get("step", 0))
                failure_done = max(failure_done, done)
                code = self.exit_code(source)
                alive = source.pid_alive(pid)
                age = time.time() - float(hb.get("ts", time.time()))
                health = source.http_status(self.port(source.spec))

                line = ""
                reason = ""
                # 进程刚死时，wrapper 可能还没来得及写 exit_code；先补一次读取
                if code is None and not alive:
                    time.sleep(0.4)
                    code = self.exit_code(source)
                if code is not None and code not in (0, 42):
                    line = self.find_error_line(source)
                    reason = f"进程退出 rc={code}"
                elif not alive:
                    line = self.find_error_line(source)
                    reason = "进程消失"
                elif hb and age > self.job.hang_s:
                    reason = f"心跳 {age:.1f}s 未更新"
                elif health == 0 and hb and age > 2.0:
                    # 端口没了但进程还在（正在退出的窗口）：以心跳停摆为准
                    line = self.find_error_line(source)
                    reason = f"健康检查失败且心跳停摆 {age:.1f}s"
                elif health not in (200, 0):
                    line = self.find_error_line(source)
                    reason = f"健康检查异常 HTTP {health}"
                if reason:
                    detect_ts = time.time()
                    # 故障时刻取最后一条心跳的时间；不要把"开始监控的时间"当成故障起点
                    onset_ts = float(hb.get("ts", 0) or 0) or detect_ts
                    self.timeline.add(
                        "fault_detected", f"{reason}（已完成 {failure_done} 样本）；{line[:200]}"
                    )
                    fault = self.translate(line, source.display)
                    fault.detected_at = detect_ts
                    fault.onset_at = onset_ts
                    fault.detection_latency_s = max(0.0, detect_ts - onset_ts)
                    break
                if code == 0 and done >= self.job.total:
                    break

            if fault is None:
                raise RuntimeError("监控窗口内没有故障（可调大 --timeout 或检查注入参数）")

            self.publish_alert(fault)
            self.timeline.add(
                "translated",
                f"{fault.code.id} {fault.code.name} [{fault.severity.value}]",
                ufc=fault.code.id,
                severity=fault.severity.value,
            )

            # ---- 保存 ----
            alive = source.pid_alive(pid)
            progress = self.save_state(source, pid, alive)
            saved_done = int(progress.get("done", 0))
            rpo = max(0, failure_done - saved_done)

            # ---- 迁移 ----
            transferred = self.transfer_state(source, target)
            source.write_text(
                f"{self.job.remote_root}/runs/quarantine-{self.source_spec.name}.json",
                json.dumps(
                    {"node": self.source_spec.name, "ufc": fault.code.id, "ts": time.time()},
                    ensure_ascii=False,
                ),
            )
            self.timeline.add("quarantine", f"{source.display} 已标记隔离")

            target_pid = self.start_worker(target, resume=True)
            resumed = wait_until(
                lambda: (self.heartbeat(target) or {}).get("step", 0) > 0, timeout_s=120
            )
            if not resumed:
                raise RuntimeError("目标机未恢复进度")
            resumed_ts = time.time()
            self.timeline.add(
                "resumed",
                f"{target.display} 从 {saved_done} 继续",
                done=(self.heartbeat(target) or {}).get("step"),
            )

            verify = self.verify(target, target_pid, saved_done, source_rate)
            verified_ts = time.time()
            self.timeline.add(
                "verified" if verify.get("verified") else "verify_failed",
                f"done={verify.get('done')} integrity={verify.get('integrity')}",
            )

            if not self.keep_running:
                target.write_text(f"{self.run_dir(target.spec)}/ctl/force_exit", "1")
                self.timeline.add("target_stopped", "演示结束后停止目标机 worker")

            summary = {
                "job": self.job.name,
                "source": source.display,
                "target": target.display,
                "ufc": fault.code.id,
                "ufc_name": fault.code.name,
                "severity": fault.severity.value,
                "verified": bool(verify.get("verified")),
                "failure_done": failure_done,
                "saved_done": saved_done,
                "rpo_samples": rpo,
                "mttd_s": round(max(0.0, detect_ts - onset_ts), 2),
                "save_s": round(resumed_ts - detect_ts, 2),
                "mttr_s": round(verified_ts - detect_ts, 2),
                "total_s": round(verified_ts - started, 1),
                "source_rate": round(source_rate, 2),
                "target_rate": verify.get("target_rate"),
                "integrity": verify.get("integrity"),
                "transferred": transferred,
                "alert": self.alert,
                "verify": verify,
            }
            self.result = summary
            self.write_artifacts(source, target, summary)
            return summary
        finally:
            source.close()
            target.close()

    def write_artifacts(self, source: RemoteHost, target: RemoteHost, summary: dict) -> None:
        (self.out_dir / "timeline.jsonl").write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in self.timeline.rows) + "\n",
            encoding="utf-8",
        )
        (self.out_dir / "result.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        lines = [
            f"# 推理任务跨机迁移：{summary['job']}",
            "",
            f"- 故障：`{summary['ufc']}` {summary['ufc_name']}（{summary['severity']}）",
            f"- 源机：{summary['source']} → 目标机：{summary['target']}",
            f"- 故障时已完成：{summary['failure_done']} 样本；保存点：{summary['saved_done']}",
            f"- 丢失样本（RPO）：{summary['rpo_samples']}",
            f"- 检测时延（MTTD）：{summary['mttd_s']}s；保存耗时：{summary['save_s']}s",
            f"- 恢复时延（MTTR）：{summary['mttr_s']}s；总耗时：{summary['total_s']}s",
            f"- 源机吞吐：{summary['source_rate']} samples/s；目标机吞吐：{summary['target_rate']} samples/s",
            f"- 完整性：{summary['integrity']}",
            f"- 验证：{'通过' if summary['verified'] else '失败'}",
            "",
            "## 时间线",
            "",
            "| 时间 | 事件 | 详情 |",
            "| --- | --- | --- |",
        ]
        for row in self.timeline.rows:
            lines.append(f"| {row['ts']} | {row['kind']} | {str(row.get('detail',''))[:120]} |")
        (self.out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="推理任务跨机迁移演示")
    parser.add_argument("--config", required=True, help="含 hosts/job/fault 的 YAML")
    parser.add_argument("--out", default="results/inference-migration")
    parser.add_argument("--keep-running", action="store_true", help="验证后不停止目标机 worker")
    args = parser.parse_args(argv)
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    out = Path(args.out)
    if out.exists():
        import shutil

        shutil.rmtree(out, ignore_errors=True)
    controller = InferenceMigration(config, out, keep_running=args.keep_running)
    result = controller.run()
    print("\n===== 结果 =====")
    print(json.dumps({k: v for k, v in result.items() if k not in ("alert", "verify")},
                     ensure_ascii=False, indent=2))
    return 0 if result.get("verified") else 1


if __name__ == "__main__":
    raise SystemExit(main())
