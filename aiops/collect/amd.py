"""AMD 采集器：``rocm-smi --json``、``--showretiredpages`` 与 amdgpu 内核日志。

AMD 的 JSON 字段在不同 ROCm 版本间差异很大（有的键带单位、有的直接嵌套），
所以这里用「别名 + 宽松数值解析」，缺字段就跳过而不是报错。
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping

from ..faultcode.models import Target, Vendor, VendorEvent
from .base import Sample, Snapshot, num

# 键别名 -> 统一指标名。顺序即优先级。
FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "gpu.temp_c": (
        "Temperature (Sensor edge) (C)",
        "Temperature (Sensor junction) (C)",
        "Temperature (Sensor memory) (C)",
        "Temperature (C)",
    ),
    "gpu.power_w": (
        "Average Graphics Package Power (W)",
        "Current Socket Graphics Package Power (W)",
        "Power (W)",
    ),
    "gpu.util": ("GPU use (%)", "GPU Use (%)"),
    "gpu.mem_used_ratio": ("GPU Memory Allocated (VRAM%)", "VRAM%"),
    "gpu.mem_used_mib": ("VRAM Total Used Memory (B)",),
    "gpu.mem_total_mib": ("VRAM Total Memory (B)",),
    "gpu.throttle_active": ("GPU Throttle Status",),
    "gpu.ecc_corrected_total": ("ECC CORRECTABLE",),
    "gpu.ecc_uncorrected": ("ECC UNCORRECTABLE",),
}

BYTE_KEYS = {"gpu.mem_used_mib", "gpu.mem_total_mib"}

_AMD_FAULT_PATTERNS = {
    "RING_TIMEOUT": r"amdgpu.*ring.*timeout",
    "GPU_RESET": r"gpu reset (begin|fail)|reset may be required|needs reset",
    "RAS_UE": r"(uncorrectable|ue).*(error|ras)|ras.*uncorrectable",
    "XGMI_ERROR": r"xgmi.*error",
    "DEVICE_LOST": r"device lost",
    "PAGE_FAULT": r"(page fault|gpu fault detected|vm_l2)",
}


def _iter_cards(payload: Mapping) -> Iterable[tuple[str, Mapping]]:
    """兼容 ``{"card0": {...}}`` 与 ``{"system": {"card0": {...}}}`` 两种顶层结构。"""

    if "system" in payload and isinstance(payload["system"], Mapping):
        payload = payload["system"]  # type: ignore[assignment]
    for key, value in payload.items():
        if isinstance(value, Mapping) and str(key).startswith(("card", "GPU")):
            yield str(key), value


def _pick(card: Mapping, aliases: Iterable[str]) -> float | None:
    for alias in aliases:
        if alias in card:
            return num(card[alias])
    return None


class AmdCollector:
    name = "amd"

    def __init__(self, node: str, source: str | Path | None = None, timeout: float = 10.0):
        self.node = node
        self.source = source
        self.timeout = timeout
        self.last_error: str = ""

    def _read_json(self) -> str:
        if self.source is None:
            try:
                out = subprocess.run(
                    ["rocm-smi", "--showallinfo", "--json"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.last_error = f"rocm-smi 调用失败: {exc}"
                return ""
            if out.returncode != 0:
                self.last_error = f"rocm-smi 退出码 {out.returncode}"
                return ""
            return out.stdout
        path = Path(self.source)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return str(self.source)

    def parse(self, text: str) -> list[Sample]:
        if not text.strip():
            return []
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            self.last_error = f"JSON 解析失败: {exc}"
            return []
        ts = time.time()
        samples: list[Sample] = []
        for card_key, card in _iter_cards(payload):
            metrics: dict[str, float] = {}
            for metric, aliases in FIELD_ALIASES.items():
                value = _pick(card, aliases)
                if value is None:
                    continue
                if metric in BYTE_KEYS:
                    value = value / (1024 * 1024)
                metrics[metric] = value
            if metrics.get("gpu.mem_used_mib") and metrics.get("gpu.mem_total_mib"):
                metrics["gpu.mem_used_ratio"] = (
                    metrics["gpu.mem_used_mib"] / metrics["gpu.mem_total_mib"]
                )
            metrics.setdefault("gpu.util", 0.0)
            metrics.setdefault("gpu.health", 1.0)
            metrics.setdefault("gpu.throttle_active", 0.0)
            samples.append(
                Sample(
                    ts=ts,
                    kind="gpu",
                    target=Target(
                        node=self.node,
                        gpu=card_key.replace("card", ""),
                        vendor=Vendor.AMD,
                    ),
                    metrics=metrics,
                    labels={
                        "vendor": "amd",
                        "card": card_key,
                        "name": str(card.get("Card Series") or card.get("Card model") or ""),
                        "pci": str(card.get("PCI Bus") or ""),
                    },
                )
            )
        return samples

    def collect(self) -> Snapshot:
        text = self._read_json()
        samples = self.parse(text)
        snap = Snapshot(ts=time.time(), samples=samples, source_up=bool(samples))
        if not samples and self.last_error:
            snap.notes.append(self.last_error)
        return snap

    @staticmethod
    def parse_retired_pages(text: str) -> dict[str, float]:
        """``rocm-smi --showretiredpages`` 输出。"""

        out: dict[str, float] = {}
        for line in text.splitlines():
            m = re.search(r"GPU\[(\d+)\].*?Retired pages count\s*:\s*(\d+)", line)
            if m:
                out[m.group(1)] = float(m.group(2))
        return out

    @staticmethod
    def parse_kernel_lines(lines: Iterable[str], node: str) -> list[VendorEvent]:
        events: list[VendorEvent] = []
        for line in lines:
            low = line.lower()
            for code, pattern in _AMD_FAULT_PATTERNS.items():
                if re.search(pattern, low):
                    events.append(
                        VendorEvent(
                            vendor=Vendor.AMD, code=code, text=line.strip(), component=node
                        )
                    )
                    break
        return events
