"""从真实设备输出派生测试 fixture。

为什么需要它：``nvidia-smi -q -x`` 的结构随驱动版本变化，用真实输出做基线最稳；
而 ECC / 行重映射这些字段在消费级卡（RTX 4090/5060 无 ECC）上恒为 N/A，
只能把真实 XML 改写成 A100 档位的字段值，用于验证解析与检测路径。

用法::

    python scripts/make_fixtures.py
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

FIXTURES = Path(__file__).resolve().parent.parent / "aiops" / "collect" / "fixtures"


def _set(gpu: ET.Element, path: str, value: str) -> None:
    """按路径写值；字段缺失时创建（驱动版本间字段会增减）。"""

    node = gpu.find(path)
    if node is not None:
        node.text = value
        return
    parent = gpu
    parts = path.split("/")
    for part in parts[:-1]:
        child = parent.find(part)
        if child is None:
            child = ET.SubElement(parent, part)
        parent = child
    ET.SubElement(parent, parts[-1]).text = value


def make_a100_ecc() -> Path:
    src = FIXTURES / "nvidia_smi_q.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    gpu = root.find("gpu")
    assert gpu is not None
    _set(gpu, "product_name", "NVIDIA A100-SXM4-80GB")
    _set(gpu, "product_architecture", "Ampere")
    _set(gpu, "ecc_mode/current_ecc", "Enabled")
    _set(gpu, "ecc_mode/pending_ecc", "Enabled")
    for scope in ("volatile", "aggregate"):
        for field in (
            "sram_correctable",
            "sram_uncorrectable_parity",
            "sram_uncorrectable_secded",
            "dram_correctable",
            "dram_uncorrectable",
        ):
            _set(gpu, f"ecc_errors/{scope}/{field}", "0")
    _set(gpu, "ecc_errors/aggregate/sram_threshold_exceeded", "No")
    _set(gpu, "ecc_errors/aggregate/dram_correctable", "37")
    _set(gpu, "ecc_errors/volatile/dram_uncorrectable", "1")
    _set(gpu, "ecc_errors/channel_repair_pending", "Yes")
    _set(gpu, "ecc_errors/unrepairable_memory", "No")
    # 行重映射：真实 RTX 输出是单个 N/A 节点，这里整体替换
    old = gpu.find("remapped_rows")
    if old is not None:
        gpu.remove(old)
    remap = ET.SubElement(gpu, "remapped_rows")
    for name, text in (
        ("correctable", "0"),
        ("uncorrectable", "0"),
        ("pending", "2"),
        ("failure", "No"),
        ("remapping_failure", "No"),
    ):
        ET.SubElement(remap, name).text = text
    _set(gpu, "retired_pages/multiple_single_bit_retirement/retired_count", "2")
    _set(gpu, "retired_pages/double_bit_retirement/retired_count", "1")
    _set(gpu, "temperature/gpu_temp", "91 C")
    _set(gpu, "temperature/gpu_temp_max_tlimit_threshold", "95 C")
    _set(gpu, "gpu_power_readings/instant_power_draw", "382.10 W")
    _set(gpu, "gpu_power_readings/power_management_limit", "400.00 W")
    _set(gpu, "fb_memory_usage/total", "81920 MiB")
    _set(gpu, "fb_memory_usage/used", "61000 MiB")
    _set(gpu, "fb_memory_usage/free", "20920 MiB")
    _set(gpu, "pci/pci_gpu_link_info/link_widths/current_link_width", "16x")
    _set(gpu, "pci/pci_gpu_link_info/pcie_gen/current_link_gen", "4")
    _set(gpu, "pci/replay_counter", "128")
    ET.indent(tree, space="  ")
    out = FIXTURES / "nvidia_smi_q_a100_ecc.xml"
    tree.write(out, encoding="utf-8", xml_declaration=True)
    return out


if __name__ == "__main__":
    path = make_a100_ecc()
    print(f"written: {path} ({path.stat().st_size} bytes)")
