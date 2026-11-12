"""可移植的 DDP 启动器（不依赖 torchrun 的 elastic agent）。

为什么需要它：``torchrun --standalone`` 在 Windows 的 PyTorch 构建上依赖 libuv +
TCPStore，本机无法启动；而 4090 机器上用它最省事。为了"两种环境同一套代码"，
这里用 ``FileStore`` 做 rendezvous：不占端口、不需要 libuv，Linux/Windows 都能跑。

生产环境仍推荐 torchrun（弹性 rendezvous、故障重启、节点加入退出）；本启动器
只用于演示与实验，接口与 torchrun 对齐（RANK / WORLD_SIZE / LOCAL_RANK 环境变量）。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from demo import protocol


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--store-path", default="")
    known, rest = parser.parse_known_args(argv)

    attempt = protocol.next_attempt_name(known.run_dir)
    attempt_dir = protocol.hb_root(known.run_dir) / attempt
    attempt_dir.mkdir(parents=True, exist_ok=True)
    protocol.set_current_attempt(known.run_dir, attempt)

    store_path = known.store_path or str(Path(known.run_dir) / f"filestore-{attempt}")
    Path(store_path).parent.mkdir(parents=True, exist_ok=True)
    Path(store_path).unlink(missing_ok=True)

    env = dict(os.environ)
    children: list[subprocess.Popen] = []
    for rank in range(known.world_size):
        child_env = dict(env)
        child_env.update(
            {
                "RANK": str(rank),
                "WORLD_SIZE": str(known.world_size),
                "LOCAL_RANK": str(rank),
                "AIOPS_ATTEMPT": attempt.split("-")[-1],
            }
        )
        argv_child = [
            sys.executable,
            "-m",
            "demo.torch_ddp_trainer",
            *rest,
            "--run-dir",
            known.run_dir,
            "--store-path",
            store_path,
        ]
        children.append(
            subprocess.Popen(argv_child, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        )

    rc = 0
    for child in children:
        out, _ = child.communicate()
        if child.returncode not in (0, None):
            rc = child.returncode
            sys.stderr.write(f"[torch-launcher] rank 退出码 {child.returncode}\n")
            if out:
                sys.stderr.write(out[-4000:])
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
