"""真实 PyTorch DDP 训练器：用同一套控制协议验证"作业级检查点重启"。

和 ``sim_trainer`` 的区别只有一处——它真的跑 GPU/CPU 计算与 NCCL/Gloo 集合通信；
心跳、控制文件、检查点格式完全一致，所以同一个 Supervisor 不需要改一行代码。

用法（2 卡）::

    torchrun --nproc_per_node 2 --standalone -m demo.torch_ddp_trainer \
        --job demo --steps 200 --ckpt-dir runs/demo/ckpt --run-dir runs/demo

注意：torchrun 自己负责 rendezvous 与 rank 环境变量；本文件不启动子进程。
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import signal
import sys
import time
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

from aiops.supervisor.checkpoint import CheckpointStore
from demo import protocol

EXIT_HARDWARE = 9
EXIT_GRACEFUL = 42


class TinyNet(nn.Module):
    def __init__(self, width: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(128, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 4)
        )

    def forward(self, x):  # noqa: D401
        return self.net(x)


def _beats_dir(run_dir: str, attempt: str) -> Path:
    root = protocol.hb_root(run_dir)
    path = root / f"attempt-{attempt}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", default="torch-demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--control-dir", default="")
    parser.add_argument("--ckpt-every", type=int, default=20)
    parser.add_argument("--node", default="localhost")
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--exclude-rank", action="append", default=[])
    parser.add_argument(
        "--store-path",
        default="",
        help="用 FileStore 代替 TCPStore 做 rendezvous（无 torchrun 时使用，Windows 上也不需要 libuv）",
    )
    args = parser.parse_args(argv)

    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if str(rank) in args.exclude_rank:
        return 0

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"
    if use_cuda:
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")
    # 超时设短：真实故障里 NCCL 默认 30 分钟才发现 rank 掉线，控制面等不起
    if args.store_path:
        Path(args.store_path).parent.mkdir(parents=True, exist_ok=True)
        store = dist.FileStore(args.store_path, world)
        dist.init_process_group(
            backend=backend,
            store=store,
            world_size=world,
            rank=rank,
            timeout=timedelta(seconds=30),
        )
    else:
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=30))

    torch.manual_seed(1234)
    model = TinyNet().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None)
    optim = torch.optim.SGD(model.parameters(), lr=args.lr)

    store = CheckpointStore(args.ckpt_dir, args.job, keep=3)
    attempt = os.environ.get("AIOPS_ATTEMPT", "1")
    beats = _beats_dir(args.run_dir, attempt)
    heartbeat = beats / f"rank-{rank}.json"
    if rank == 0:
        protocol.set_current_attempt(args.run_dir, f"attempt-{attempt}")

    control = Path(args.control_dir) if args.control_dir else Path(args.run_dir) / "ctl"
    control.mkdir(parents=True, exist_ok=True)

    start_step = 0
    if args.resume:
        latest = store.latest()
        if latest is not None:
            state = torch.load(Path(latest.path) / "state.pt", map_location=device, weights_only=False)
            model.load_state_dict(state["model"])
            optim.load_state_dict(state["optim"])
            start_step = int(state.get("step", latest.step))

    # 看门狗：即使 rank 卡在 allreduce 里，也能在强制退出时立刻死掉，
    # 避免控制面杀了 launcher 却留下持有 GPU 的孤儿进程。
    def _watchdog() -> None:
        while True:
            if (control / "force_exit").exists():
                os._exit(130)
            time.sleep(0.3)

    threading.Thread(target=_watchdog, daemon=True).start()
    stop_flag = {"value": False}

    def _on_term(signum, frame):  # noqa: ANN001
        stop_flag["value"] = True

    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _on_term)

    def beat(step: int, step_time: float, loss: float, state: str = "running", **extra) -> None:
        payload = {
            "rank": rank,
            "world_size": world,
            "step": step,
            "step_time_s": round(step_time, 4),
            "loss": loss,
            "state": state,
            "node": args.node,
            "ts": time.time(),
            "pid": os.getpid(),
            **extra,
        }
        tmp = heartbeat.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        try:
            os.replace(tmp, heartbeat)
        except PermissionError:
            pass

    def ctl(name: str) -> str | None:
        path = control / name
        if not path.exists():
            return None
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return "1"

    def save(step: int) -> None:
        def _write(path: Path) -> None:
            torch.save({"model": model.state_dict(), "optim": optim.state_dict(), "step": step}, path / "state.pt")

        if rank == 0:
            store.save(step, _write, world_size=world, samples=step * args.batch, kind="train")
        dist.barrier()

    def finish(step: int, code: int, state: str = "done") -> int:
        if state == "done" and rank == 0:
            pass
        beat(step, 0.0, 0.0, state=state)
        dist.barrier()
        dist.destroy_process_group()
        return code

    beat(start_step, 0.0, 0.0, resumed=bool(args.resume))
    torch.manual_seed(2000 + start_step)

    for step in range(start_step + 1, args.steps + 1):
        if ctl(f"die.{rank}") is not None:
            beat(step, 0.0, 0.0, state="failed", reason="injected hardware fault")
            dist.destroy_process_group()
            return EXIT_HARDWARE
        if ctl("force_exit") is not None:
            beat(step, 0.0, 0.0, state="failed", reason="force exit")
            try:
                dist.destroy_process_group()
            except Exception:
                pass
            return 130
        if ctl("graceful") is not None or stop_flag["value"]:
            save(step - 1)
            return finish(step - 1, EXIT_GRACEFUL, state="done")
        if ctl(f"hang.{rank}") is not None:
            beat(step, 0.0, 0.0, state="running", hung=True)
            while ctl(f"hang.{rank}") is not None:
                time.sleep(0.2)

        t0 = time.perf_counter()
        throttle = ctl(f"throttle.{rank}")
        if throttle:
            try:
                time.sleep(float(throttle) * 0.005)
            except ValueError:
                pass
        x = torch.randn(args.batch, 128, device=device)
        y = torch.randint(0, 4, (args.batch,), device=device)
        loss = nn.functional.cross_entropy(model(x), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        step_time = time.perf_counter() - t0
        beat(step, step_time, float(loss.detach().cpu()))

        if args.ckpt_every > 0 and step % args.ckpt_every == 0:
            save(step)

    save(args.steps)
    return finish(args.steps, 0, state="done")


if __name__ == "__main__":
    sys.exit(main())
