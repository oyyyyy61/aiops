"""RTX 4090 上的真实基线测量：步耗时、显存、检查点开销。

只依赖 torch，不启动控制面。用于给场景实验提供"这台机器的正常值"：
没有基线，"慢"就没有定义。

用法::

    python -m experiments.baseline_4090 --steps 60 --out results/4090-baseline.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from aiops.supervisor.checkpoint import CheckpointStore


class Net(nn.Module):
    def __init__(self, width: int, depth: int, in_dim: int = 128):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, width), nn.ReLU()]
        for _ in range(max(0, depth - 1)):
            layers += [nn.Linear(width, width), nn.ReLU()]
        layers.append(nn.Linear(width, 4))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def measure(width: int, depth: int, batch: int, steps: int, warmup: int, device: torch.device) -> dict:
    model = Net(width, depth).to(device)
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    params = sum(p.numel() for p in model.parameters())
    times: list[float] = []
    for step in range(warmup + steps):
        t0 = time.perf_counter()
        x = torch.randn(batch, 128, device=device)
        y = torch.randint(0, 4, (batch,), device=device)
        loss = nn.functional.cross_entropy(model(x), y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if step >= warmup:
            times.append(dt)
    return {
        "width": width,
        "depth": depth,
        "batch": batch,
        "params_m": round(params / 1e6, 2),
        "step_ms_p50": round(statistics.median(times) * 1000, 2),
        "step_ms_p95": round(sorted(times)[int(len(times) * 0.95) - 1] * 1000, 2),
        "step_ms_min": round(min(times) * 1000, 2),
        "step_ms_max": round(max(times) * 1000, 2),
        "steps_per_s": round(1 / statistics.median(times), 1),
    }


def measure_checkpoint(width: int, depth: int, out_dir: Path, repeats: int = 3) -> dict:
    model = Net(width, depth)
    optim = torch.optim.SGD(model.parameters(), lr=0.05)
    store = CheckpointStore(out_dir, "bench", keep=repeats + 1)
    saves, loads, sizes = [], [], []
    for i in range(repeats):
        t0 = time.perf_counter()

        def _write(path: Path, i=i) -> None:
            torch.save(
                {"model": model.state_dict(), "optim": optim.state_dict(), "step": i},
                path / "state.pt",
            )

        meta = store.save(i, _write, world_size=1)
        saves.append(time.perf_counter() - t0)
        sizes.append(Path(meta.path, "state.pt").stat().st_size)
        t1 = time.perf_counter()
        state = torch.load(Path(meta.path) / "state.pt", weights_only=False)
        _ = state["model"]
        loads.append(time.perf_counter() - t1)
    return {
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "size_mb": round(statistics.median(sizes) / 1024**2, 1),
        "save_ms_p50": round(statistics.median(saves) * 1000, 1),
        "load_ms_p50": round(statistics.median(loads) * 1000, 1),
        "repeats": repeats,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--out", default="results/4090-baseline.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    props = torch.cuda.get_device_properties(0) if device.type == "cuda" else None
    result = {
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "total_memory_gib": round(props.total_memory / 1024**3, 1) if props else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "shapes": [],
        "checkpoints": [],
    }
    print(json.dumps({k: v for k, v in result.items() if k != "shapes"}, ensure_ascii=False))

    for width, depth in ((512, 3), (1024, 4), (2048, 4)):
        stats = measure(width, depth, args.batch, args.steps, args.warmup, device)
        result["shapes"].append(stats)
        print(json.dumps(stats, ensure_ascii=False), flush=True)

    for width, depth in ((1024, 4), (2048, 4)):
        ckpt = measure_checkpoint(width, depth, Path(args.out).parent / "ckpt-bench")
        result["checkpoints"].append(ckpt)
        print(json.dumps(ckpt, ensure_ascii=False), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
