"""NVIDIA 采集器：``nvidia-smi -q -x`` XML 与 dmesg/Xid 解析。

不依赖 pynvml：XML 是每台机器都有的最稳接口，字段变化时解析器向前兼容。
"""

from __future__ import annotations

import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

from ..faultcode.models import Target, Vendor, VendorEvent
from .base import Sample, Snapshot, boolish, num

# Xid 语义摘要（用于事件证据，不替代 UFC 目录）
XID_SUMMARY = {
    13: "graphics engine exception",
    31: "illegal memory access / page fault",
    43: "GPU stopped processing",
    48: "double-bit ECC error",
    62: "internal micro-controller halt",
    63: "ECC page retirement / row remap recorded",
    64: "ECC page retirement / row remap failure",
    74: "NVLink error",
    79: "GPU has fallen off the bus",
    92: "high single-bit ECC error rate",
    94: "contained ECC error",
    95: "uncontained ECC error",
    109: "context switch timeout",
    119: "GSP RPC timeout",
    120: "GSP error",
    140: "unrecovered ECC error",
}


def _t(node: ET.Element | None, path: str) -> str | None:
    if node is None:
        return None
    found = node.find(path)
    if found is None:
        return None
    return (found.text or "").strip()


def _f(node: ET.Element | None, path: str, default: float | None = None) -> float | None:
    return num(_t(node, path), default)


class NvidiaCollector:
    """从 XML 文件、字符串或实时命令采集单机所有 GPU。"""

    name = "nvidia"

    def __init__(self, node: str, source: str | Path | None = None, timeout: float = 10.0):
        self.node = node
        self.source = source
        self.timeout = timeout
        self.last_error: str = ""

    # ---------- 原始读取 ----------

    def _read_xml(self) -> str:
        if self.source is None:
            try:
                out = subprocess.run(
                    ["nvidia-smi", "-q", "-x"],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                self.last_error = f"nvidia-smi 调用失败: {exc}"
                return ""
            if out.returncode != 0:
                self.last_error = f"nvidia-smi 退出码 {out.returncode}: {out.stderr.strip()[:200]}"
                return ""
            return out.stdout
        path = Path(self.source)
        if path.exists():
            return path.read_text(encoding="utf-8", errors="replace")
        # 允许直接传 XML 文本（测试/回放）
        return str(self.source)

    # ---------- 解析 ----------

    def parse(self, xml_text: str) -> list[Sample]:
        if not xml_text.strip():
            return []
        try:
            root = ET.fromstring(xml_text)
        except ET.ParseError as exc:
            self.last_error = f"XML 解析失败: {exc}"
            return []
        driver = _t(root, "driver_version") or ""
        ts = time.time()
        samples: list[Sample] = []
        for gpu in root.findall("gpu"):
            gpu_id = gpu.get("id") or _t(gpu, "pci_bus_id") or ""
            uuid = _t(gpu, "uuid") or ""
            idx = _t(gpu, "minor_number") or ""
            # minor_number 在 WDDM 下常为 N/A，用序号兜底
            label = idx if idx and idx.upper() != "N/A" else str(len(samples))
            metrics: dict[str, float] = {}
            self._fill_identity(metrics, gpu)
            self._fill_util(metrics, gpu)
            self._fill_temp_power(metrics, gpu)
            self._fill_memory(metrics, gpu)
            self._fill_ecc(metrics, gpu)
            self._fill_remap(metrics, gpu)
            self._fill_pcie(metrics, gpu)
            self._fill_throttle(metrics, gpu)
            metrics["gpu.health"] = self._health(metrics)
            samples.append(
                Sample(
                    ts=ts,
                    kind="gpu",
                    target=Target(node=self.node, gpu=label, vendor=Vendor.NVIDIA),
                    metrics=metrics,
                    labels={
                        "vendor": "nvidia",
                        "gpu_id": gpu_id,
                        "uuid": uuid,
                        "name": _t(gpu, "product_name") or "",
                        "driver": driver,
                        "arch": _t(gpu, "product_architecture") or "",
                    },
                )
            )
        return samples

    # ---------- 字段装配 ----------

    def _fill_identity(self, m: dict[str, float], gpu: ET.Element) -> None:
        m["gpu.vbios_ok"] = 1.0 if _t(gpu, "vbios_version") else 0.0

    def _fill_util(self, m: dict[str, float], gpu: ET.Element) -> None:
        util = _f(gpu, "utilization/gpu_util", 0.0)
        mem_util = _f(gpu, "utilization/memory_util", 0.0)
        m["gpu.util"] = util if util is not None else 0.0
        m["gpu.mem_util"] = mem_util if mem_util is not None else 0.0

    def _fill_temp_power(self, m: dict[str, float], gpu: ET.Element) -> None:
        temp = _f(gpu, "temperature/gpu_temp")
        if temp is not None:
            m["gpu.temp_c"] = temp
        shutdown = _f(gpu, "temperature/gpu_temp_max_threshold")
        if shutdown is not None:
            m["gpu.temp_limit_c"] = shutdown
        m["gpu.power_w"] = _f(gpu, "gpu_power_readings/instant_power_draw", 0.0) or 0.0
        m["gpu.power_limit_w"] = _f(gpu, "gpu_power_readings/power_management_limit", 0.0) or 0.0
        cap = m.get("gpu.power_limit_w") or _f(gpu, "gpu_power_readings/enforced_power_limit", 0.0)
        if cap:
            m["gpu.power_ratio"] = min(1.0, m["gpu.power_w"] / cap) if cap > 0 else 0.0

    def _fill_memory(self, m: dict[str, float], gpu: ET.Element) -> None:
        total = _f(gpu, "fb_memory_usage/total")
        used = _f(gpu, "fb_memory_usage/used")
        if total:
            m["gpu.mem_total_mib"] = total
            m["gpu.mem_used_mib"] = used or 0.0
            m["gpu.mem_used_ratio"] = (used or 0.0) / total

    def _fill_ecc(self, m: dict[str, float], gpu: ET.Element) -> None:
        m["gpu.ecc_enabled"] = boolish(_t(gpu, "ecc_mode/current_ecc") or "N/A")
        unc = 0.0
        for path in (
            "ecc_errors/volatile/dram_uncorrectable",
            "ecc_errors/volatile/sram_uncorrectable_parity",
            "ecc_errors/volatile/sram_uncorrectable_secded",
        ):
            unc += _f(gpu, path, 0.0) or 0.0
        m["gpu.ecc_uncorrected"] = unc
        corr = _f(gpu, "ecc_errors/aggregate/dram_correctable", 0.0) or 0.0
        m["gpu.ecc_corrected_total"] = corr
        m["gpu.ecc_corrected_trip"] = boolish(_t(gpu, "ecc_errors/aggregate/sram_threshold_exceeded") or "No")
        m["gpu.ecc_channel_repair_pending"] = boolish(
            _t(gpu, "ecc_errors/channel_repair_pending") or "No"
        )
        m["gpu.unrepairable_memory"] = boolish(_t(gpu, "ecc_errors/unrepairable_memory") or "No")

    def _fill_remap(self, m: dict[str, float], gpu: ET.Element) -> None:
        node = gpu.find("remapped_rows")
        pending = 0.0
        failure = 0.0
        if node is not None and (node.text or "").strip().upper() != "N/A":
            pending = _f(gpu, "remapped_rows/pending", 0.0) or 0.0
            failure = boolish(_t(gpu, "remapped_rows/failure") or "No")
        if not failure:
            failure = float(boolish(_t(gpu, "ecc_errors/channel_repair_pending") or "No"))
        m["gpu.row_remap_pending"] = pending
        m["gpu.row_remap_failure"] = failure

    def _fill_pcie(self, m: dict[str, float], gpu: ET.Element) -> None:
        m["gpu.pcie_replay_total"] = _f(gpu, "pci/replay_counter", 0.0) or 0.0
        m["gpu.pcie_link_gen"] = _f(gpu, "pci/pci_gpu_link_info/pcie_gen/current_link_gen", 0.0) or 0.0
        m["gpu.pcie_link_gen_max"] = _f(gpu, "pci/pci_gpu_link_info/pcie_gen/max_link_gen", 0.0) or 0.0
        m["gpu.pcie_link_width"] = _f(
            gpu, "pci/pci_gpu_link_info/link_widths/current_link_width", 0.0
        ) or 0.0

    def _fill_throttle(self, m: dict[str, float], gpu: ET.Element) -> None:
        reasons = gpu.find("clocks_throttle_reasons")
        active = 0.0
        thermal = 0.0
        power = 0.0
        if reasons is not None:
            mapping = {
                "hw_thermal_slowdown": "thermal",
                "sw_thermal_slowdown": "thermal",
                "hw_power_brake_slowdown": "power",
                "sw_power_cap": "power",
                "hw_slowdown": "power",
            }
            for child in reasons:
                if boolish(child.text or "Not Active") <= 0:
                    continue
                name = child.tag.lower()
                kind = mapping.get(name)
                active += 1.0
                if kind == "thermal":
                    thermal += 1.0
                elif kind == "power":
                    power += 1.0
        m["gpu.throttle_active"] = active
        m["gpu.thermal_throttle"] = thermal
        m["gpu.power_brake"] = power

    def _health(self, m: dict[str, float]) -> float:
        if m.get("gpu.ecc_uncorrected", 0) > 0:
            return 0.0
        if m.get("gpu.row_remap_failure", 0) > 0:
            return 0.0
        limit = m.get("gpu.temp_limit_c")
        if limit and m.get("gpu.temp_c", 0) >= limit:
            return 0.0
        return 1.0

    # ---------- 采集入口 ----------

    def collect(self) -> Snapshot:
        text = self._read_xml()
        samples = self.parse(text)
        snap = Snapshot(ts=time.time(), samples=samples, source_up=bool(samples))
        if not samples and self.last_error:
            snap.notes.append(self.last_error)
        return snap

    # ---------- Xid ----------

    @staticmethod
    def parse_xid_lines(lines: Iterable[str], node: str) -> list[VendorEvent]:
        """从 dmesg / 容器日志抽取 Xid 事件。

        真实行样例::

            NVRM: Xid (PCI:0000:2d:00): 79, GPU has fallen off the bus.
        """

        import re

        pattern = re.compile(r"Xid\s*\(([^)]*)\)\s*:\s*(\d+)", re.IGNORECASE)
        events: list[VendorEvent] = []
        for line in lines:
            m = pattern.search(line)
            if not m:
                continue
            code = m.group(2)
            events.append(
                VendorEvent(
                    vendor=Vendor.NVIDIA,
                    code=code,
                    text=line.strip(),
                    component=m.group(1),
                    raw={"xid": int(code), "summary": XID_SUMMARY.get(int(code), "")},
                )
            )
        return events


def read_xid_from_dmesg(node: str, timeout: float = 10.0) -> list[VendorEvent]:
    """读取最近的 dmesg 并抽取 Xid（Linux）。"""

    try:
        out = subprocess.run(
            ["dmesg", "-T"], capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if out.returncode != 0:
        return []
    return NvidiaCollector.parse_xid_lines(out.stdout.splitlines()[-4000:], node)
