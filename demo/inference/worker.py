"""推理 worker：可续跑、可迁移的在线/离线推理服务。

设计目标不是"跑个模型"，而是让"保存进度 → 换机器续跑"有真实的语义：

* **可续跑的进度**：每处理完一个 batch 就原子更新 ``progress.json``，结果逐行追加到
  ``results.jsonl``；恢复时按 ``done`` 截断结果文件，避免重复/缺失。
* **可观测**：``/health`` 与 ``/metrics`` 暴露进度、队列、批延迟；心跳文件供控制面读取。
* **可注入真实错误**：``device-assert`` 用越界 token 触发**真实的 CUDA device-side
  assert**（不是伪造日志），``cuda-oom`` 触发真实的 OOM，便于验证"错误码翻译"链路。

启动（由控制面通过 SSH 拉起）::

    python -m demo.inference.worker --job infer --total 2000 --port 18000 \
        --state-dir runs/infer/state --run-dir runs/infer --control-dir runs/infer/ctl \
        --node node-a --resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch
import torch.nn as nn

EXIT_OK = 0
EXIT_FAULT = 3
EXIT_GRACEFUL = 42
EXIT_FORCED = 130


# ---------------------------------------------------------------- 模型


class TinyLM(nn.Module):
    """小号 decoder-only 模型：embedding + 2 层 Transformer + tied lm_head。"""

    def __init__(self, vocab: int = 32000, dim: int = 256, layers: int = 2, heads: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab, dim)
        block = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4, batch_first=True, dropout=0.0
        )
        self.encoder = nn.TransformerEncoder(block, num_layers=layers)
        self.lm_head = nn.Linear(dim, vocab, bias=False)
        self.lm_head.weight = self.embed.weight  # tied

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.embed(ids))
        return self.lm_head(h[:, -1, :])


# ---------------------------------------------------------------- 状态


def _atomic_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class State:
    """进度 + 结果文件的一致性维护。"""

    def __init__(self, state_dir: Path, job: str):
        self.dir = state_dir
        self.job = job
        self.progress_path = state_dir / "progress.json"
        self.results_path = state_dir / "results.jsonl"
        self.ckpt_dir = state_dir / "ckpt"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

    @property
    def done(self) -> int:
        data = _read_json(self.progress_path) or {}
        return int(data.get("done", 0))

    def load(self, total: int, node: str, resume: bool) -> int:
        if not resume:
            if self.results_path.exists():
                self.results_path.unlink()
            _atomic_write(
                self.progress_path,
                {"job": self.job, "done": 0, "total": total, "node": node, "ts": time.time()},
            )
            return 0
        data = _read_json(self.progress_path)
        if not data:
            return 0
        done = int(data.get("done", 0))
        # 结果文件可能比进度多（上次崩溃在两者之间），截断到 done 行保证一致
        if self.results_path.exists():
            lines = self.results_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > done:
                self.results_path.write_text(
                    "\n".join(lines[:done]) + ("\n" if done else ""), encoding="utf-8"
                )
        return done

    def append(self, rows: list[dict]) -> None:
        with self.results_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def commit(self, done: int, total: int, node: str, every: int) -> None:
        _atomic_write(
            self.progress_path,
            {"job": self.job, "done": done, "total": total, "node": node, "ts": time.time()},
        )
        if every > 0 and done % every == 0:
            _atomic_write(
                self.ckpt_dir / f"ckpt-{done:06d}.json",
                {"job": self.job, "done": done, "total": total, "node": node, "ts": time.time()},
            )


# ---------------------------------------------------------------- HTTP


class _Handler(BaseHTTPRequestHandler):
    worker: "Worker"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            payload = {
                "ok": self.worker.alive,
                "node": self.worker.args.node,
                "job": self.worker.args.job,
                "done": self.worker.done,
                "total": self.worker.args.total,
                "state": self.worker.state_name,
            }
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200 if self.worker.alive else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/metrics"):
            m = self.worker.metrics()
            body = "\n".join(f"{k} {v}" for k, v in m.items()).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args) -> None:  # 静默
        return


# ---------------------------------------------------------------- Worker


class Worker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.node = args.node
        self.run_dir = Path(args.run_dir)
        self.ctl = Path(args.control_dir) if args.control_dir else self.run_dir / "ctl"
        self.ctl.mkdir(parents=True, exist_ok=True)
        self.state = State(Path(args.state_dir), args.job)
        self.done = 0
        self.state_name = "starting"
        self.alive = True
        self.batch_ms: list[float] = []
        self.last_batch_ts = 0.0
        self.started_ts = time.time()
        self.device = torch.device(args.device)
        self._stop = {"flag": False}

    # ---- 控制面 ----

    def _ctl(self, name: str) -> bool:
        return (self.ctl / name).exists()

    def _install_signals(self) -> None:
        def _handler(signum, frame):  # noqa: ANN001
            self._stop["flag"] = True

        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                signal.signal(sig, _handler)
            except (ValueError, OSError):
                pass

    # ---- 心跳 ----

    def _hb_path(self) -> Path:
        attempt = os.environ.get("AIOPS_ATTEMPT", "1")
        root = self.run_dir / "hb" / f"attempt-{attempt}"
        root.mkdir(parents=True, exist_ok=True)
        return root / "rank-0.json"

    def beat(self, step_time: float = 0.0, forward_s: float = 0.0, state: str | None = None) -> None:
        if state:
            self.state_name = state
        payload = {
            "rank": 0,
            "world_size": 1,
            "step": self.done,
            "step_time_s": round(step_time, 4),
            "forward_s": round(forward_s, 5),
            "loss": 0.0,
            "state": self.state_name,
            "node": self.node,
            "ts": time.time(),
            "pid": os.getpid(),
            "total": self.args.total,
        }
        _atomic_write(self._hb_path(), payload)

    def metrics(self) -> dict[str, float]:
        recent = self.batch_ms[-20:] or [0.0]
        return {
            "inference_done_samples": float(self.done),
            "inference_total_samples": float(self.args.total),
            "inference_pending_samples": float(max(0, self.args.total - self.done)),
            "inference_batch_latency_ms": round(sum(recent) / len(recent), 2),
            "inference_alive": 1.0 if self.alive else 0.0,
        }

    # ---- 主循环 ----

    def run(self) -> int:
        self._install_signals()
        self.done = self.state.load(self.args.total, self.node, self.args.resume)
        print(
            f"[worker] node={self.node} job={self.args.job} resume={self.args.resume} "
            f"from={self.done}/{self.args.total}",
            flush=True,
        )
        self.beat(state="running")

        server = None
        if self.args.port:
            _Handler.worker = self
            try:
                server = ThreadingHTTPServer(("127.0.0.1", self.args.port), _Handler)
                threading.Thread(target=server.serve_forever, daemon=True).start()
            except OSError as exc:
                print(f"[worker] 健康检查端口 {self.args.port} 不可用: {exc}", flush=True)
                server = None

        torch.manual_seed(7)
        model = TinyLM(self.args.vocab, self.args.dim, self.args.layers).to(self.device).eval()
        batch = self.args.batch
        exit_code = EXIT_OK
        try:
            with torch.inference_mode():
                while self.done < self.args.total:
                    if self._ctl("force_exit"):
                        self.beat(state="stopped")
                        exit_code = EXIT_FORCED
                        break
                    if self._stop["flag"] or self._ctl("graceful"):
                        self.beat(state="stopped")
                        exit_code = EXIT_GRACEFUL
                        break
                    if self._ctl("die"):
                        self.beat(state="failed")
                        exit_code = 9
                        break
                    if self._ctl("hang"):
                        self.beat(state="hung")
                        while self._ctl("hang"):
                            time.sleep(0.2)
                        self.beat(state="running")

                    if (
                        self.args.fault == "device-assert"
                        and self.done >= self.args.fault_at > 0
                    ):
                        # 真实 CUDA device-side assert：越界 token 索引
                        bad = torch.full((1, 4), self.args.vocab + 1, dtype=torch.long, device=self.device)
                        print("[worker] injecting out-of-range token id -> CUDA device assert", flush=True)
                        model(bad)
                        torch.cuda.synchronize()

                    if self.args.fault == "cuda-oom" and self.done >= self.args.fault_at > 0:
                        print("[worker] injecting oversized allocation -> CUDA OOM", flush=True)
                        torch.empty(1 << 34, device=self.device)

                    t0 = time.perf_counter()
                    ids = self._tokens(self.done, batch)
                    logits = model(ids)
                    forward_s = time.perf_counter() - t0
                    next_ids = logits.argmax(dim=-1).detach().cpu().tolist()
                    rows = [
                        {"i": self.done + k, "next": int(next_ids[k]), "node": self.node}
                        for k in range(len(next_ids))
                    ]
                    self.state.append(rows)
                    self.done += len(rows)
                    self.state.commit(self.done, self.args.total, self.node, self.args.ckpt_every)
                    elapsed = time.perf_counter() - t0
                    self.batch_ms.append(elapsed * 1000)
                    self.last_batch_ts = time.time()
                    self.beat(step_time=elapsed, forward_s=forward_s, state="running")
                    if self.args.qps > 0:
                        due = t0 + len(rows) / self.args.qps
                        gap = due - time.perf_counter()
                        if gap > 0:
                            time.sleep(gap)
        except Exception as exc:  # noqa: BLE001
            # 把 torch/驱动抛出的原始错误打出来，控制面靠这行做故障码翻译
            print(f"[worker] FATAL {type(exc).__name__}: {exc}", flush=True)
            self.beat(state="failed")
            exit_code = EXIT_FAULT
        finally:
            self.alive = False
            if server is not None:
                server.shutdown()
            self._write_exit_code(exit_code)
            print(f"[worker] exit code={exit_code} done={self.done}", flush=True)
        return exit_code

    def _tokens(self, start: int, batch: int) -> torch.Tensor:
        """按样本下标确定性生成输入，保证迁移前后同一批样本的输入一致。"""

        gen = torch.Generator(device="cpu").manual_seed(1000 + start)
        ids = torch.randint(0, min(self.args.vocab, 4096), (batch, self.args.seq), generator=gen)
        return ids.to(self.device)

    def _write_exit_code(self, code: int) -> None:
        try:
            (self.run_dir / "exit_code").write_text(str(code), encoding="utf-8")
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="可续跑的推理 worker")
    parser.add_argument("--job", default="infer")
    parser.add_argument("--total", type=int, default=2000)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--seq", type=int, default=64)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--vocab", type=int, default=32000)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--control-dir", default="")
    parser.add_argument("--node", default="node-a")
    parser.add_argument("--port", type=int, default=18000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt-every", type=int, default=128)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fault", choices=["none", "device-assert", "cuda-oom"], default="none")
    parser.add_argument("--fault-at", type=int, default=0)
    parser.add_argument("--qps", type=float, default=0.0, help="限速：每秒处理的样本数（模拟在线流量）")
    args = parser.parse_args(argv)
    return Worker(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
