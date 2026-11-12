"""仿真训练器的观测探针：把心跳/检查点读成统一的 Snapshot + JobStatus。

这份文件是"采集层"在仿真侧的对应物。换成真机时，只需要把这一层换掉：
真实作业的 step_time 来自 torch 的 step 计时（或 Prometheus 的
``training_step_seconds``），其余检测、决策、执行逻辑完全不动。
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from aiops.collect.base import Sample, Snapshot
from aiops.faultcode.models import Target, Vendor
from aiops.supervisor.checkpoint import CheckpointStore
from aiops.supervisor.supervisor import JobStatus, Probe
from demo import protocol


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class HeartbeatProbe:
    """读取仿真 rank 的心跳，产出 Supervisor 需要的 Probe。"""

    def __init__(self, run_dir: str | Path, store: CheckpointStore, job: str, node: str, world_size: int):
        self.run_dir = Path(run_dir)
        self.store = store
        self.job = job
        self.node = node
        self.world_size = world_size
        self._last_step = 0
        self._last_progress_ts = time.time()

    def __call__(self) -> Probe:
        now = time.time()
        hb_dir = protocol.current_attempt_dir(self.run_dir)
        beats: dict[str, dict] = {}
        if hb_dir.exists():
            for path in hb_dir.glob("rank-*.json"):
                data = _read_json(path)
                if data:
                    beats[str(data.get("rank"))] = data

        steps = [b.get("step", 0) for b in beats.values()]
        step = min(steps) if steps else 0
        step_times = {
            rank: float(b.get("step_time_s", 0.0)) for rank, b in beats.items()
            if b.get("state") == "running" and b.get("step_time_s", 0) > 0
        }
        worst_step_time = max(step_times.values()) if step_times else 0.0
        state = "running"
        if beats and all(b.get("state") == "done" for b in beats.values()):
            state = "done"
        elif any(b.get("state") == "failed" for b in beats.values()):
            state = "failed"
        elif any(b.get("state") == "done" for b in beats.values()):
            state = "partial_done"

        if step > self._last_step:
            self._last_step = step
            self._last_progress_ts = now
        progress_age = now - self._last_progress_ts

        latest_ckpt = self.store.latest()
        ckpt_age = latest_ckpt.age_s(now) if latest_ckpt else 0.0
        losses = [b.get("loss", 0.0) for b in beats.values() if b.get("state") == "running"]
        loss = max(losses) if losses else 0.0
        loss_nan = 1.0 if any(isinstance(x, float) and x != x for x in losses) else 0.0
        step_skew = (max(step_times.values()) / max(1e-9, sorted(step_times.values())[len(step_times)//2])) if len(step_times) >= 2 else 1.0

        metrics = {
            "job.step": float(step),
            "job.step_time_s": worst_step_time,
            "job.progress_age_s": progress_age,
            "job.loss": loss if loss == loss else 0.0,
            "job.loss_nan": loss_nan,
            "job.ckpt_age_s": ckpt_age,
            "job.ckpt_fail": 0.0,
            "job.step_skew_ratio": float(step_skew),
            "job.rank_alive_ratio": len([b for b in beats.values() if b.get("state") == "running"]) / max(1, self.world_size),
        }
        sample = Sample(
            ts=now,
            kind="job",
            target=Target(node=self.node, job=self.job),
            metrics=metrics,
            labels={"vendor": "generic", "state": state},
        )
        status = JobStatus(
            running=state == "running",
            rc=None,
            step=step,
            step_time_s=worst_step_time,
            step_times=step_times,
            progress_age_s=progress_age,
            state=state,
            node=self.node,
            world_size=self.world_size,
            loss=loss,
        )
        return Probe(snapshot=Snapshot(ts=now, samples=[sample]), status=status)
