"""昇腾采集器：``npu-smi info`` / ``info -t health|usages|board`` 解析。

昇腾没有 NVIDIA 那样稳定的 XML 接口，输出是表格（``npu-smi info``）与键值对
（``-t health`` / ``-t usages``）混合，且版本间字段名有差异。这里同时支持两条解析
路径，并且只依赖"看到什么解析什么"，缺字段不影响其它指标。
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Iterable, Mapping

from ..faultcode.models import Target, Vendor, VendorEvent
from .base import Sample, Snapshot, num

HEALTH_MAP = {
    "OK": 1.0,
    "NORMAL": 1.0,
    "WARNING": 0.5,
    "ALARM": 0.0,
    "CRITICAL": 0.0,
    "0": 1.0,
}

FIELD_MAP = {
    "power(w)": "gpu.power_w",
    "power-usage(w)": "gpu.power_w",
    "temp(c)": "gpu.temp_c",
    "temperature(c)": "gpu.temp_c",
    "aicore(%)": "gpu.util",
    "aicpu(%)": "gpu.aicpu_util",
    "hbm-usage(mb)": "gpu.mem_used_mib",
    "memory-usage(mb)": "gpu.mem_used_mib",
    "hbm capacity(mb)": "gpu.mem_total_mib",
}


class AscendCollector:
    name = "ascend"

    def __init__(self, node: str, source: str | Path | None = None, timeout: float = 10.0):
        self.node = node
        self.source = source
        self.timeout = timeout
        self.last_error: str = ""

    # ---------- 读取 ----------

    def _read_text(self) -> str:
        if self.source is None:
            try:
                out = subprocess.run(
                    ["npu-smi", "info"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.last_error = f"npu-smi 调用失败: {exc}"
                return ""
            if out.returncode != 0:
                self.last_error = f"npu-smi 退出码 {out.returncode}"
                return ""
            return out.stdout
        path = Path(self.source)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        return str(self.source)

    # ---------- 键值型 ----------

    def parse_health(self, text: str) -> tuple[str, str]:
        """返回 (health, error_code|detail)。"""

        health = re.search(r"Health\s*Status\s*:\s*(\w+)", text, re.IGNORECASE)
        err = re.search(r"Error\s*Code\s*:\s*([0-9A-Fa-fx]+)", text, re.IGNORECASE)
        info = re.search(r"Error\s*Information\s*:\s*(.+)", text, re.IGNORECASE)
        status = (health.group(1) if health else "OK").upper()
        code = err.group(1) if err else "0"
        detail = info.group(1).strip() if info else ""
        suffix = f"|{detail}" if detail and detail.lower() != "none" else ""
        return status, code + suffix

    def parse_usages(self, text: str) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, _, raw = line.partition(":")
            metric = FIELD_MAP.get(key.strip().lower())
            if not metric:
                continue
            if "/" in raw:
                used_s, _, total_s = raw.partition("/")
                used, total = num(used_s), num(total_s)
                if used is not None and total:
                    metrics[metric] = used
                    metrics["gpu.mem_total_mib"] = total
                    metrics["gpu.mem_used_mib"] = used
                    metrics["gpu.mem_used_ratio"] = used / total
                continue
            value = num(raw.strip())
            if value is not None:
                metrics[metric] = value
        return metrics

    # ---------- 表格型 npu-smi info ----------

    def parse_table(self, text: str) -> list[Sample]:
        """解析 ``npu-smi info``。

        每个 NPU 占两行：主行（NPU/Name、Health、Power/Temp）与续行
        （Chip、Bus-Id、AICore/Memory/HBM）。这里用续行状态机合并。
        """

        samples: list[Sample] = []
        current: dict[str, object] | None = None
        ts = time.time()
        for line in text.splitlines():
            if "|" not in line or "===" in line:
                continue
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) < 3:
                continue
            head = cells[0].split()
            if not head or not head[0].isdigit():
                continue
            metrics: dict[str, float] = {}
            right = cells[2]
            if len(head) >= 2:  # 主行：NPU ID + 型号
                health = cells[1].upper().split()[0] if cells[1] else "OK"
                metrics["gpu.health"] = HEALTH_MAP.get(health, 1.0)
                m = re.search(r"(\d+(?:\.\d+)?)\s+(\d{2,3})\s+(\d+)\s*/\s*(\d+)", right)
                if m:
                    metrics["gpu.power_w"] = float(m.group(1))
                    metrics["gpu.temp_c"] = float(m.group(2))
                current = {
                    "npu": head[0],
                    "name": head[1],
                    "metrics": metrics,
                    "bus": "",
                }
                samples.append(
                    Sample(
                        ts=ts,
                        kind="gpu",
                        target=Target(node=self.node, gpu=head[0], vendor=Vendor.ASCEND),
                        metrics=metrics,
                        labels={"vendor": "ascend", "name": head[1]},
                    )
                )
                continue
            if current is None:  # 续行：Chip ID -> Bus-Id / 使用率
                continue
            current["bus"] = cells[1]
            m = re.search(
                r"(\d+)\s+(\d+)\s*/\s*(\d+)\s+(\d+)\s*/\s*(\d+)", right
            )
            if m:
                aicore, mem_used, mem_total, hbm_used, hbm_total = (int(g) for g in m.groups())
                target_metrics = current["metrics"]  # type: ignore[assignment]
                target_metrics["gpu.util"] = float(aicore)
                used, total = (hbm_used, hbm_total) if hbm_total > 0 else (mem_used, mem_total)
                if total > 0:
                    target_metrics["gpu.mem_used_mib"] = float(used)
                    target_metrics["gpu.mem_total_mib"] = float(total)
                    target_metrics["gpu.mem_used_ratio"] = used / total
        for sample in samples:
            sample.metrics.setdefault("gpu.util", 0.0)
        return samples

    def collect(self) -> Snapshot:
        text = self._read_text()
        samples = self.parse_table(text)
        snap = Snapshot(ts=time.time(), samples=samples, source_up=bool(samples))
        if not samples and self.last_error:
            snap.notes.append(self.last_error)
        return snap

    # ---------- 事件 ----------

    @staticmethod
    def parse_event_lines(lines: Iterable[str], node: str) -> list[VendorEvent]:
        events: list[VendorEvent] = []
        for line in lines:
            low = line.lower()
            health = re.search(r"Health\s*Status\s*:\s*(\w+)", line, re.IGNORECASE)
            if health and health.group(1).upper() not in {"OK", "NORMAL"}:
                events.append(
                    VendorEvent(
                        vendor=Vendor.ASCEND,
                        code=f"HEALTH_{health.group(1).upper()}",
                        text=line.strip(),
                        component=node,
                    )
                )
                continue
            for code, pattern in (
                ("HBM_ECC_UNCORRECTED", r"hbm.*ecc"),
                ("AI_CORE_EXCEPTION", r"aicore.*(exception|error)"),
                ("HCCS_LINK_DOWN", r"hccs.*(down|error|fail)"),
                ("DEVICE_LOST", r"device\s*lost"),
                ("HCCL_TIMEOUT", r"hccl.*timeout"),
                ("DRIVER_ERROR", r"davinci.*(error|fail)"),
            ):
                if re.search(pattern, low):
                    events.append(
                        VendorEvent(
                            vendor=Vendor.ASCEND, code=code, text=line.strip(), component=node
                        )
                    )
                    break
        return events


def read_npu_health(node: str, npu_id: int = 0, timeout: float = 10.0) -> tuple[str, str]:
    """调用 ``npu-smi info -t health -i N``，返回 (status, code)。"""

    try:
        out = subprocess.run(
            ["npu-smi", "info", "-t", "health", "-i", str(npu_id)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "UNKNOWN", "tool-unavailable"
    if out.returncode != 0:
        return "UNKNOWN", f"rc={out.returncode}"
    return AscendCollector(node).parse_health(out.stdout)
