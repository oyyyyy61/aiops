"""在 vLLM 上跑的推理任务 worker（可续跑、可迁移、可做一致性校验）。

和 ``worker.py`` 的区别：

* 真正用 **vLLM** 起一个 OpenAI 兼容的推理引擎（``vllm serve`` 子进程），
  控制面通过 HTTP 提交请求；
* 每次启动都会跑一组**固定校准题**（算术），用来对比迁移前后的"效果"；
* 恢复时会**重放**若干已完成样本，逐字比对，检测"跨卡/跨节点结果偏移"；
* 故障注入 ``engine-kill``：直接杀掉 vLLM 引擎进程，模拟推理引擎崩溃。

状态文件与 ``worker.py`` 一致（progress.json / results.jsonl），迁移逻辑完全复用。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PY = sys.executable
EXIT_OK = 0
EXIT_FAULT = 3
EXIT_GRACEFUL = 42
EXIT_FORCED = 130

CALIBRATION = [
    ("3+5=?", "8"),
    ("9-4=?", "5"),
    ("6*7=?", "42"),
    ("12+13=?", "25"),
    ("100-37=?", "63"),
    ("8*8=?", "64"),
    ("45+55=?", "100"),
    ("7*9=?", "63"),
    ("81/9=?", "9"),
    ("14+28=?", "42"),
]


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


def prompt_of(i: int) -> str:
    """确定性地生成第 i 条请求：只要求模型输出一个整数，便于判定效果。"""

    a, b = i % 37 + 2, i % 19 + 3
    return f"请只回答数字，不要解释：{a}+{b}=?"


def expected_of(i: int) -> int:
    a, b = i % 37 + 2, i % 19 + 3
    return a + b


def extract_int(text: str) -> int | None:
    """取文本里**最后一个**整数："3 + 5 = 8" -> 8。"""

    import re

    nums = re.findall(r"\d+", text)
    if not nums:
        return None
    try:
        return int(nums[-1])
    except ValueError:
        return None


class State:
    def __init__(self, state_dir: Path, job: str):
        self.dir = state_dir
        self.job = job
        self.progress_path = state_dir / "progress.json"
        self.results_path = state_dir / "results.jsonl"
        self.checks_path = state_dir / "checks.json"
        self.dir.mkdir(parents=True, exist_ok=True)

    @property
    def done(self) -> int:
        return int((_read_json(self.progress_path) or {}).get("done", 0))

    def load(self, total: int, node: str, resume: bool) -> int:
        if not resume:
            if self.results_path.exists():
                self.results_path.unlink()
            if self.checks_path.exists():
                self.checks_path.unlink()
            _atomic_write(
                self.progress_path,
                {"job": self.job, "done": 0, "total": total, "node": node, "ts": time.time()},
            )
            return 0
        data = _read_json(self.progress_path)
        if not data:
            return 0
        done = int(data.get("done", 0))
        if self.results_path.exists():
            lines = self.results_path.read_text(encoding="utf-8").splitlines()
            if len(lines) > done:
                self.results_path.write_text(
                    "\n".join(lines[:done]) + ("\n" if done else ""), encoding="utf-8"
                )
        return done

    def rows(self) -> list[dict]:
        if not self.results_path.exists():
            return []
        out = []
        for line in self.results_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
        return out

    def append(self, row: dict) -> None:
        with self.results_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    def commit(self, done: int, total: int, node: str) -> None:
        _atomic_write(
            self.progress_path,
            {"job": self.job, "done": done, "total": total, "node": node, "ts": time.time()},
        )

    def record_check(self, node: str, probe: dict, replay: dict) -> None:
        data = _read_json(self.checks_path) or {"history": []}
        data.setdefault("history", []).append(
            {"node": node, "ts": time.time(), "probe": probe, "replay": replay}
        )
        _atomic_write(self.checks_path, data)


class Engine:
    """vLLM 子进程 + HTTP 客户端。"""

    def __init__(self, model: str, port: int, gpu_mem: float = 0.45, log_path: Path | None = None):
        self.model = model
        self.port = port
        self.gpu_mem = gpu_mem
        self.proc: subprocess.Popen | None = None
        self.log_path = log_path

    def start(self, timeout: float = 300.0) -> bool:
        log = open(self.log_path, "ab") if self.log_path else subprocess.DEVNULL
        # vLLM 的 flashinfer JIT 会调用 ninja；非交互 SSH 的 PATH 里往往没有 conda bin
        env = dict(os.environ)
        env["PATH"] = "/root/miniconda3/bin:" + env.get("PATH", "")
        # 首次启动时 flashinfer 会用 ninja 现编 sampling 核，耗时几分钟；
        # 演示/迁移场景关掉它，改用 PyTorch 采样（对 temperature=0 的结果无影响）。
        env.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self.proc = subprocess.Popen(
            [
                PY, "-m", "vllm.entrypoints.openai.api_server",
                "--model", self.model,
                "--served-model-name", "demo",
                "--port", str(self.port),
                "--gpu-memory-utilization", str(self.gpu_mem),
                "--max-model-len", "2048",
                "--enforce-eager",
                "--disable-log-requests",
            ],
            stdout=log,
            stderr=log,
            env=env,
            start_new_session=True,   # 让 EngineCore 与 api_server 同组，便于整组清理
        )
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            if self.healthy():
                return True
            time.sleep(2)
        return False

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{self.port}/health", timeout=3) as resp:
                return resp.status == 200
        except (urllib.error.URLError, OSError):
            return False

    def complete(self, prompt: str, max_tokens: int = 16, timeout: float = 60.0) -> str:
        # 用 chat 接口套 instruct 模板：Qwen2.5-Instruct 的算术正确率才可用来衡量"效果"
        body = json.dumps(
            {
                "model": "demo",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "seed": 0,
            }
        ).encode()
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return str(data["choices"][0]["message"]["content"])

    def _cleanup_children(self) -> None:
        # api_server 会再拉起 EngineCore；按进程组整组清理最稳
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                pass
        subprocess.run(
            ["pkill", "-9", "-f", f"vllm.entrypoints.openai.api_server.*--port {self.port}"],
            check=False,
            capture_output=True,
        )

    def kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.kill()
            except OSError:
                pass
        self._cleanup_children()

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=30)
            except (OSError, subprocess.TimeoutExpired):
                self.kill()
        self._cleanup_children()


class _Handler(BaseHTTPRequestHandler):
    worker: "VllmWorker"

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/health"):
            payload = {
                "ok": self.worker.engine_ok,
                "node": self.worker.args.node,
                "job": self.worker.args.job,
                "done": self.worker.done,
                "total": self.worker.args.total,
                "state": self.worker.state_name,
            }
            body = json.dumps(payload, ensure_ascii=False).encode()
            self.send_response(200 if self.worker.engine_ok else 503)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path.startswith("/metrics"):
            body = "\n".join(
                f"{k} {v}" for k, v in self.worker.metrics().items()
            ).encode() + b"\n"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def log_message(self, *args) -> None:
        return


class VllmWorker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.node = args.node
        self.run_dir = Path(args.run_dir)
        self.ctl = Path(args.control_dir) if args.control_dir else self.run_dir / "ctl"
        self.ctl.mkdir(parents=True, exist_ok=True)
        self.state = State(Path(args.state_dir), args.job)
        self.done = 0
        self.state_name = "starting"
        self.engine_ok = False
        self.engine = Engine(args.model, args.vllm_port, args.gpu_mem, self.run_dir / "engine.log")
        self.latencies: list[float] = []
        self._stop = {"flag": False}

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

    def _hb_path(self) -> Path:
        attempt = os.environ.get("AIOPS_ATTEMPT", "1")
        root = self.run_dir / "hb" / f"attempt-{attempt}"
        root.mkdir(parents=True, exist_ok=True)
        return root / "rank-0.json"

    def beat(self, step_time: float = 0.0, forward_s: float = 0.0, state: str | None = None) -> None:
        if state:
            self.state_name = state
        _atomic_write(
            self._hb_path(),
            {
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
                "engine_ok": self.engine_ok,
            },
        )

    def metrics(self) -> dict[str, float]:
        recent = self.latencies[-20:] or [0.0]
        return {
            "inference_done_samples": float(self.done),
            "inference_total_samples": float(self.args.total),
            "inference_pending_samples": float(max(0, self.args.total - self.done)),
            "inference_request_latency_ms": round(sum(recent) / len(recent), 2),
            "inference_engine_ok": 1.0 if self.engine_ok else 0.0,
        }

    # ---- 一致性检查 ----

    def calibration(self) -> dict:
        hits = 0
        samples = []
        for prompt, want in CALIBRATION:
            try:
                text = self.engine.complete(prompt, max_tokens=24)
            except Exception as exc:  # noqa: BLE001
                return {"ok": False, "error": str(exc)[:200], "accuracy": 0.0}
            got = extract_int(text)
            hit = got is not None and got == int(want)
            hits += 1 if hit else 0
            samples.append({"prompt": prompt, "want": int(want), "got": text.strip()[:40], "hit": hit})
        return {
            "ok": True,
            "accuracy": round(hits / len(CALIBRATION), 3),
            "hits": hits,
            "samples": samples[:3],
        }

    def probe(self) -> dict:
        text = self.engine.complete("请只输出一个词：你好", max_tokens=4)
        digest = hashlib.sha256(text.strip().encode()).hexdigest()[:12]
        return {"text": text.strip()[:40], "hash": digest}

    def replay(self, sample_size: int = 5) -> dict:
        rows = self.state.rows()
        if not rows:
            return {"checked": 0, "matched": 0}
        step = max(1, len(rows) // sample_size)
        picked = rows[::step][:sample_size]
        matched = 0
        mismatched: list[dict] = []
        for row in picked:
            try:
                text = self.engine.complete(prompt_of(int(row["i"])), max_tokens=16)
            except Exception as exc:  # noqa: BLE001
                return {"checked": len(picked), "matched": matched, "error": str(exc)[:200]}
            if text.strip() == str(row.get("text", "")).strip():
                matched += 1
            else:
                mismatched.append({"i": row["i"], "old": row.get("text", "")[:40], "new": text.strip()[:40]})
        return {"checked": len(picked), "matched": matched, "mismatched": mismatched}

    # ---- 主流程 ----

    def run(self) -> int:
        self._install_signals()
        self.done = self.state.load(self.args.total, self.node, self.args.resume)
        print(f"[vllm-worker] node={self.node} resume={self.args.resume} from={self.done}/{self.args.total}", flush=True)
        self.beat(state="starting")

        server = None
        try:
            server = ThreadingHTTPServer(("127.0.0.1", self.args.port), _Handler)
            _Handler.worker = self
            threading.Thread(target=server.serve_forever, daemon=True).start()
        except OSError as exc:
            print(f"[vllm-worker] 控制端口不可用: {exc}", flush=True)

        ok = self.engine.start()
        if not ok:
            print("[vllm-worker] FATAL vLLM engine failed to start", flush=True)
            self.engine_ok = False
            self.beat(state="failed")
            self._write_exit_code(EXIT_FAULT)
            self.engine.stop()
            return EXIT_FAULT
        self.engine_ok = True
        self.beat(state="running")

        probe = self.probe()
        calib = self.calibration()
        replay = self.replay()
        self.state.record_check(self.node, {**probe, "calibration": calib}, replay)
        print(
            f"[vllm-worker] probe={probe['hash']} calib={calib.get('accuracy')} "
            f"replay={replay.get('matched')}/{replay.get('checked')}",
            flush=True,
        )

        exit_code = EXIT_OK
        try:
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
                    self.engine.kill()
                    self.engine_ok = False
                    self.beat(state="failed")
                    exit_code = 9
                    break
                if (
                    self.args.fault == "engine-kill"
                    and self.args.fault_at > 0
                    and self.done >= self.args.fault_at
                ):
                    print("[vllm-worker] injecting engine-kill: SIGKILL vLLM child", flush=True)
                    self.engine.kill()
                    self.engine_ok = False
                    # 引擎已死，接下来的请求会失败；把真实错误打出来给控制面翻译
                    self.engine.complete(prompt_of(self.done))

                t0 = time.perf_counter()
                text = self.engine.complete(prompt_of(self.done), max_tokens=16)
                latency = time.perf_counter() - t0
                self.latencies.append(latency * 1000)
                row = {"i": self.done, "text": text.strip(), "node": self.node, "latency_ms": round(latency * 1000, 2)}
                self.state.append(row)
                self.done += 1
                self.state.commit(self.done, self.args.total, self.node)
                self.beat(step_time=latency, forward_s=latency, state="running")
                if self.args.qps > 0:
                    due = t0 + 1.0 / self.args.qps
                    gap = due - time.perf_counter()
                    if gap > 0:
                        time.sleep(gap)
        except Exception as exc:  # noqa: BLE001
            print(f"[vllm-worker] FATAL {type(exc).__name__}: {exc}", flush=True)
            self.beat(state="failed")
            exit_code = EXIT_FAULT
        finally:
            self.engine_ok = False
            self.engine.stop()
            if server is not None:
                server.shutdown()
            self._write_exit_code(exit_code)
            print(f"[vllm-worker] exit code={exit_code} done={self.done}", flush=True)
        return exit_code

    def _write_exit_code(self, code: int) -> None:
        try:
            (self.run_dir / "exit_code").write_text(str(code), encoding="utf-8")
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="vLLM 推理任务 worker")
    parser.add_argument("--job", default="vllm-demo")
    parser.add_argument("--model", required=True)
    parser.add_argument("--total", type=int, default=120)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--vllm-port", type=int, default=18200)
    parser.add_argument("--gpu-mem", type=float, default=0.45)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--control-dir", default="")
    parser.add_argument("--node", default="node-a")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fault", choices=["none", "engine-kill", "die"], default="none")
    parser.add_argument("--fault-at", type=int, default=0)
    parser.add_argument("--qps", type=float, default=0.0, help="每秒请求数（用于控制演示节奏）")
    args = parser.parse_args(argv)
    return VllmWorker(args).run()


if __name__ == "__main__":
    raise SystemExit(main())
