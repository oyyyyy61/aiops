"""无依赖的仿真训练器：用来在没有真机时把控制闭环跑通、跑快、跑出数据。

它不是 mock 掉一切：进程是真的、检查点是真的、rank 死亡/卡死/变慢是真的，
只有"GPU 计算"被换成了 sleep。这样验的是控制面（检测、决策、保存、重启、验证），
而不是 PyTorch 本身——后者在 `demo/torch_ddp_trainer.py` 里做真机对照。

协议（全部通过文件系统，跨平台、可观测）：

* 心跳：``<run>/hb/attempt-<n>/rank-<i>.json``，含 step / step_time / loss / state
* 控制：``<ctl>/graceful``、``throttle.<rank>``、``hang.<rank>``、``die.<rank>``
* 检查点：``<ckpt>/<job>/ckpt-00000020/state.json``（由 CheckpointStore 原子写）

启动方式（launcher 模式，supervisor 只管理一个进程）：

    python -m demo.sim_trainer --run-dir runs/x --ckpt-dir runs/x/ckpt --steps 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

from aiops.supervisor.checkpoint import CheckpointStore
from demo import protocol

EXIT_OK = 0
EXIT_COLLECTIVE_TIMEOUT = 3
EXIT_HARDWARE = 9
EXIT_GRACEFUL = 42


# ---------------------------------------------------------------- 工具


def _write_json(path: Path, data: dict) -> None:
    """原子写心跳。

    Windows 上如果另一端（探针）正好打开着目标文件，``os.replace`` 会抛
    ``PermissionError``。这里做有限重试，避免把观测行为变成对被测进程的干扰。
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.002 + 0.001 * attempt)
    # 最终失败也不要拖垮训练进程：留下 .tmp，下一拍会覆盖
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


# ---------------------------------------------------------------- 单 rank


class Rank:
    def __init__(self, args: argparse.Namespace, rank: int, world_size: int):
        self.args = args
        self.rank = rank
        self.world_size = world_size
        self.run_dir = Path(args.run_dir)
        self.ctl = Path(args.control_dir) if args.control_dir else self.run_dir / "ctl"
        self.heartbeat = protocol.current_attempt_dir(self.run_dir) / f"rank-{rank}.json"
        self.store = CheckpointStore(args.ckpt_dir, args.job, keep=3)
        self.base_sleep = args.step_sleep
        self.rng = random.Random(1000 + rank)

    # ---- 控制面 ----

    def _ctl_text(self, name: str) -> str | None:
        path = self.ctl / name
        if path.exists():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                return "1"
        return None

    def _throttle(self) -> float:
        value = self._ctl_text(f"throttle.{self.rank}")
        if not value:
            return 1.0
        try:
            return max(1.0, float(value))
        except ValueError:
            return 1.0

    def _hang(self) -> bool:
        return (self.ctl / f"hang.{self.rank}").exists()

    def _die(self) -> bool:
        return (self.ctl / f"die.{self.rank}").exists()

    def _graceful(self) -> bool:
        return (self.ctl / "graceful").exists()

    # ---- 心跳 ----

    def beat(self, step: int, step_time: float, loss: float, state: str = "running", **extra) -> None:
        _write_json(
            self.heartbeat,
            {
                "rank": self.rank,
                "world_size": self.world_size,
                "step": step,
                "step_time_s": round(step_time, 4),
                "loss": loss,
                "state": state,
                "node": self.args.node,
                "ts": time.time(),
                "pid": os.getpid(),
                **extra,
            },
        )

    # ---- 集合通信屏障 ----

    def barrier(self, step: int, timeout: float | None = None) -> str:
        """等待所有存活 rank 到达 ``step``。返回 ``ok`` / ``timeout`` / ``peer_failed``。"""

        timeout = timeout if timeout is not None else self.args.barrier_timeout
        deadline = time.time() + timeout
        hb_dir = protocol.current_attempt_dir(self.run_dir)
        while time.time() < deadline:
            steps: dict[int, int] = {}
            failed = False
            for i in range(self.world_size):
                data = _read_json(hb_dir / f"rank-{i}.json")
                if data is None:
                    continue
                state = data.get("state", "running")
                if state == "failed":
                    failed = True
                    break
                if state == "done":
                    steps[i] = 10**9  # 已完成，视为到达任意屏障
                else:
                    steps[i] = int(data.get("step", -1))
            if failed:
                return "peer_failed"
            if len(steps) == self.world_size and all(s >= step for s in steps.values()):
                return "ok"
            time.sleep(0.02)
        return "timeout"

    # ---- 训练主循环 ----

    def run(self) -> int:
        start_step = 0
        if self.args.resume:
            latest = self.store.latest()
            if latest is not None:
                start_step = latest.step
        self.beat(start_step, 0.0, 0.0, state="running", resumed=bool(self.args.resume))
        if self.rank == 0:
            self._write_run_meta(start_step)

        for step in range(start_step + 1, self.args.steps + 1):
            if self._die():
                self.beat(step, 0.0, 0.0, state="failed", reason="injected hardware fault")
                return EXIT_HARDWARE
            if self._hang():
                self.beat(step, 0.0, 0.0, state="running", hung=True)
                # 卡死：不退出、不推进，等外部处理
                while self._hang():
                    time.sleep(0.1)
            if self._graceful():
                return self._save_and_exit(step)

            # 集合通信：所有 rank 必须到达上一步
            barrier = self.barrier(step - 1)
            if barrier != "ok":
                self.beat(step, 0.0, 0.0, state="failed", reason=f"collective {barrier}")
                return EXIT_COLLECTIVE_TIMEOUT

            t0 = time.perf_counter()
            multiplier = self._throttle()
            duration = self.base_sleep * multiplier * (0.9 + 0.2 * self.rng.random())
            time.sleep(duration)
            loss = 2.0 / (1 + step) + 0.01 * self.rng.random()
            if self._ctl_text("nan"):
                loss = float("nan")
            elapsed = time.perf_counter() - t0
            self.beat(step, elapsed, loss, forward_s=elapsed)

            if self.args.ckpt_every > 0 and step % self.args.ckpt_every == 0:
                self._checkpoint(step)
            if self._graceful():
                return self._save_and_exit(step)

        self.beat(self.args.steps, 0.0, 0.0, state="done")
        return EXIT_OK

    # ---- 检查点 ----

    def _checkpoint(self, step: int) -> None:
        if self.rank != 0:
            return
        self.store.save_json(
            step,
            {
                "step": step,
                "world_size": self.world_size,
                "loss": 2.0 / (1 + step),
                "node": self.args.node,
                "job": self.args.job,
            },
            world_size=self.world_size,
            samples=step * 32,
            kind="train",
        )

    def _save_and_exit(self, step: int) -> int:
        self._checkpoint(step)
        marker = self.ctl / "ckpt_ready"
        try:
            marker.write_text(str(step), encoding="utf-8")
        except OSError:
            pass
        self.beat(step, 0.0, 0.0, state="done", reason="graceful checkpoint")
        return EXIT_GRACEFUL

    def _write_run_meta(self, start_step: int) -> None:
        _write_json(
            self.run_dir / "run.json",
            {
                "job": self.args.job,
                "steps": self.args.steps,
                "world_size": self.world_size,
                "start_step": start_step,
                "node": self.args.node,
                "ts": time.time(),
            },
        )


# ---------------------------------------------------------------- launcher


def _spawn_children(args: argparse.Namespace) -> int:
    run_dir = Path(args.run_dir)
    attempt_dir = protocol.hb_root(run_dir) / protocol.next_attempt_name(run_dir)
    attempt_dir.mkdir(parents=True, exist_ok=True)
    protocol.set_current_attempt(run_dir, attempt_dir.name)

    children: list[subprocess.Popen] = []
    env = dict(os.environ)
    for rank in range(args.world_size):
        argv = [
            sys.executable,
            "-m",
            "demo.sim_trainer",
            "--rank",
            str(rank),
            "--world-size",
            str(args.world_size),
            "--run-dir",
            str(run_dir),
            "--ckpt-dir",
            str(args.ckpt_dir),
            "--job",
            args.job,
            "--steps",
            str(args.steps),
            "--step-sleep",
            str(args.step_sleep),
            "--ckpt-every",
            str(args.ckpt_every),
            "--node",
            args.node,
            "--barrier-timeout",
            str(args.barrier_timeout),
        ]
        if args.control_dir:
            argv += ["--control-dir", args.control_dir]
        if args.resume:
            argv.append("--resume")
        log_dir = Path(args.run_dir) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log = (log_dir / f"rank-{rank}.log").open("a", encoding="utf-8")
        children.append(
            subprocess.Popen(argv, env=env, cwd=args.workdir or None, stdout=log, stderr=log)
        )
    rc = 0
    for child in children:
        code = child.wait()
        rc = code if code not in (0,) else rc
    return rc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="无依赖仿真训练器")
    parser.add_argument("--rank", type=int, default=None, help="不传则以 launcher 模式启动全部 rank")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--control-dir", default="")
    parser.add_argument("--job", default="sim-job")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--step-sleep", type=float, default=0.1)
    parser.add_argument("--ckpt-every", type=int, default=20)
    parser.add_argument("--barrier-timeout", type=float, default=6.0)
    parser.add_argument("--node", default="gpu-node-15")
    parser.add_argument("--workdir", default="")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    if args.rank is None:
        return _spawn_children(args)

    rank = Rank(args, args.rank, args.world_size)
    return rank.run()


if __name__ == "__main__":
    raise SystemExit(main())
