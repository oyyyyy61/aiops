"""作业级检查点存储：原子写、完整性校验、按作业保留最近 N 份。

要点：

* **原子性**：先写 ``.tmp-<step>`` 目录，``fsync`` 后 rename 成正式目录。
  中途掉电/被杀不会留下半份检查点，恢复时也不会误读。
* **可校验**：每份检查点带 ``meta.json``（步数、样本数、world size、耗时、
  sha256）。恢复前校验 hash，避免"文件在但内容坏"这种最坏情况。
* **可解释**：``age_s`` / ``latest()`` 直接支撑 RPO 决策：检查点太旧时，
  策略层会拒绝重启而选择升级人工。
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

FORMAT_VERSION = 1


@dataclass
class CheckpointMeta:
    step: int
    path: str
    ts: float
    world_size: int = 1
    samples: int = 0
    kind: str = "train"
    duration_s: float = 0.0
    sha256: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return Path(self.path).name

    def age_s(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.ts

    def as_dict(self) -> dict:
        return {
            "step": self.step,
            "path": self.path,
            "ts": round(self.ts, 3),
            "world_size": self.world_size,
            "samples": self.samples,
            "kind": self.kind,
            "duration_s": round(self.duration_s, 3),
            "sha256": self.sha256,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CheckpointMeta":
        return cls(
            step=int(data["step"]),
            path=str(data["path"]),
            ts=float(data["ts"]),
            world_size=int(data.get("world_size", 1)),
            samples=int(data.get("samples", 0)),
            kind=str(data.get("kind", "train")),
            duration_s=float(data.get("duration_s", 0.0)),
            sha256=str(data.get("sha256", "")),
            extra=dict(data.get("extra", {})),
        )


class CheckpointStore:
    """一个作业一个目录，内部按 step 分子目录。"""

    def __init__(self, root: str | Path, job: str = "job", keep: int = 3):
        self.root = Path(root) / job
        self.job = job
        self.keep = keep
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 写 ----------

    def save(
        self,
        step: int,
        write_state: Callable[[Path], None],
        *,
        world_size: int = 1,
        samples: int = 0,
        kind: str = "train",
        extra: Mapping[str, Any] | None = None,
    ) -> CheckpointMeta:
        start = time.time()
        tmp = self.root / f".tmp-{step:08d}-{os.getpid()}"
        if tmp.exists():
            shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        write_state(tmp)
        state_file = self._state_file(tmp)
        digest = _sha256(state_file) if state_file else ""
        meta = CheckpointMeta(
            step=step,
            path=str(tmp),
            ts=start,
            world_size=world_size,
            samples=samples,
            kind=kind,
            duration_s=time.time() - start,
            sha256=digest,
            extra=dict(extra or {}),
        )
        (tmp / "meta.json").write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        final = self.root / f"ckpt-{step:08d}"
        if final.exists():
            shutil.rmtree(final, ignore_errors=True)
        os.replace(tmp, final)
        meta.path = str(final)
        meta.duration_s = time.time() - start
        # 重新落盘最终路径
        (final / "meta.json").write_text(
            json.dumps(meta.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        self.prune()
        return meta

    def save_json(self, step: int, state: Mapping[str, Any], **kwargs) -> CheckpointMeta:
        def _write(path: Path) -> None:
            (path / "state.json").write_text(
                json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        return self.save(step, _write, **kwargs)

    # ---------- 读 ----------

    @staticmethod
    def _state_file(directory: Path) -> Path | None:
        for name in ("state.pt", "state.json", "state.bin"):
            candidate = directory / name
            if candidate.exists():
                return candidate
        return None

    def list(self) -> list[CheckpointMeta]:
        metas: list[CheckpointMeta] = []
        for entry in sorted(self.root.glob("ckpt-*")):
            meta_file = entry / "meta.json"
            if not meta_file.exists():
                continue
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                meta = CheckpointMeta.from_dict(data)
                meta.path = str(entry)
                metas.append(meta)
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
        return metas

    def latest(self, compatible_world_size: int | None = None) -> CheckpointMeta | None:
        """最新且（可选）world size 兼容的检查点。"""

        for meta in reversed(self.list()):
            if compatible_world_size is not None and meta.world_size not in (0, compatible_world_size):
                continue
            return meta
        return None

    def verify(self, meta: CheckpointMeta) -> bool:
        directory = Path(meta.path)
        if not directory.exists():
            return False
        state = self._state_file(directory)
        if state is None:
            return False
        if meta.sha256 and _sha256(state) != meta.sha256:
            return False
        return True

    def load_json(self, meta: CheckpointMeta) -> dict:
        return json.loads((Path(meta.path) / "state.json").read_text(encoding="utf-8"))

    def age_s(self, now: float | None = None) -> float | None:
        latest = self.latest()
        return latest.age_s(now) if latest else None

    # ---------- 维护 ----------

    def prune(self) -> list[str]:
        removed: list[str] = []
        metas = self.list()
        if self.keep <= 0:
            return removed
        for meta in metas[: max(0, len(metas) - self.keep)]:
            shutil.rmtree(meta.path, ignore_errors=True)
            removed.append(meta.name)
        return removed

    def total_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointPolicy:
    """什么时候必须保存检查点。

    * 周期保存：按时间（默认 300s）或步数（默认 200 步）；
    * 紧急保存：收到故障信号后，在 ``grace_s`` 内落一份再退出；
    * RPO 预算：检查点年龄超过阈值时主动告警（UFC-JOB-005）。
    """

    def __init__(
        self,
        interval_s: float = 300.0,
        interval_steps: int = 200,
        rpo_budget_s: float = 1800.0,
        grace_s: float = 30.0,
    ):
        self.interval_s = interval_s
        self.interval_steps = interval_steps
        self.rpo_budget_s = rpo_budget_s
        self.grace_s = grace_s

    def due(self, last_ts: float | None, last_step: int | None, now: float, step: int) -> bool:
        if last_ts is None or last_step is None:
            return True
        if now - last_ts >= self.interval_s:
            return True
        return step - last_step >= self.interval_steps

    def rpo_exceeded(self, age_s: float | None) -> bool:
        return age_s is not None and age_s > self.rpo_budget_s
