"""双卡 DDP 的微基准：每步耗时与 all-reduce 有效带宽。

用来给"慢节点/网络退化"提供基线。由 ``demo.torch_launcher`` 以 2 rank 启动：

    python -m demo.torch_launcher --world-size 2 --run-dir runs/bench \
        --module experiments.ddp_bench --job bench
"""

from __future__ import annotations

import os
import statistics
import time

import torch
import torch.distributed as dist
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, width: int = 2048, depth: int = 4, in_dim: int = 128):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def main() -> int:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    store = dist.FileStore(os.environ["STORE"], world)
    dist.init_process_group("nccl", store=store, world_size=world, rank=rank)
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    model = Net().to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    batch = 256
    steps = 50
    times: list[float] = []
    for i in range(steps + 10):
        x = torch.randn(batch, 128, device=device)
        y = torch.randint(0, 4, (batch,), device=device)
        t0 = time.perf_counter()
        loss = nn.functional.cross_entropy(model(x), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        torch.cuda.synchronize()
        if i >= 10:
            times.append(time.perf_counter() - t0)

    # all-reduce 有效带宽（busbw）：12.86M 参数 ≈ 51MB
    params = sum(p.numel() for p in model.parameters())
    nbytes = params * 4
    buf = torch.ones(params // 4, device=device, dtype=torch.float32)
    for _ in range(5):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        dist.all_reduce(buf)
    torch.cuda.synchronize()
    ar_s = (time.perf_counter() - t0) / 20
    busbw = 2 * (world - 1) / world * nbytes / ar_s / 1e9

    if rank == 0:
        print(
            f"world={world} params={params/1e6:.2f}M "
            f"step_ms_p50={statistics.median(times)*1000:.2f} "
            f"step_ms_p95={sorted(times)[int(len(times)*0.95)-1]*1000:.2f} "
            f"allreduce_ms={ar_s*1000:.2f} busbw_gbps={busbw:.1f}"
        )
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
