"""厂商故障翻译测试：真实日志格式 -> 统一故障码。"""

from __future__ import annotations

import unittest
from pathlib import Path

from aiops.faultcode.models import Confidence, Target, Vendor
from aiops.faultcode.normalizer import Normalizer
from aiops.faultcode.taxonomy import load_catalog

FIXTURES = Path(__file__).resolve().parent.parent / "aiops" / "collect" / "fixtures"


class TranslationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.normalizer = Normalizer(cls.catalog)

    def _codes(self, vendor: Vendor, filename: str) -> list[str]:
        lines = (FIXTURES / filename).read_text(encoding="utf-8").splitlines()
        target = Target(node="node-a", vendor=vendor)
        return [t.fault.code.id for t in self.normalizer.normalize_lines(vendor, lines, target)]

    def test_nvidia_xid_mapping(self) -> None:
        codes = self._codes(Vendor.NVIDIA, "dmesg_nvidia_xid.txt")
        self.assertIn("UFC-GPU-001", codes)  # Xid 79
        self.assertIn("UFC-GPU-002", codes)  # Xid 31/13
        self.assertIn("UFC-MEM-001", codes)  # Xid 48/95
        self.assertIn("UFC-LNK-001", codes)  # Xid 74
        self.assertIn("UFC-GPU-003", codes)  # Xid 43

    def test_exact_code_beats_text_heuristic(self) -> None:
        # Xid 74 的行里也有 "Fatal"，不能因此被翻译成驱动崩溃
        line = "NVRM: Xid (PCI:0000:2d:00): 74, NVLink: Fatal error on link 3"
        results = self.normalizer.parse_log_line(Vendor.NVIDIA, line, Target(node="n"))
        self.assertEqual([r.fault.code.id for r in results], ["UFC-LNK-001"])

    def test_amd_mapping(self) -> None:
        codes = self._codes(Vendor.AMD, "dmesg_amd.txt")
        self.assertIn("UFC-GPU-008", codes)
        self.assertIn("UFC-MEM-001", codes)
        self.assertIn("UFC-LNK-005", codes)

    def test_ascend_mapping(self) -> None:
        codes = self._codes(Vendor.ASCEND, "slog_ascend.txt")
        self.assertIn("UFC-NET-001", codes)
        self.assertIn("UFC-MEM-001", codes)
        self.assertIn("UFC-GPU-007", codes)
        self.assertIn("UFC-LNK-003", codes)

    def test_generic_mapping_applies_to_all_vendors(self) -> None:
        line = "HCCL timeout, waiting for collective op"
        results = self.normalizer.parse_log_line(Vendor.ASCEND, line, Target(node="n"))
        self.assertEqual([r.fault.code.id for r in results], ["UFC-NET-001"])

    def test_unknown_code_is_recorded_not_dropped(self) -> None:
        normalizer = Normalizer(self.catalog)
        results = normalizer.translate_raw(Vendor.NVIDIA, "Xid 999")
        self.assertEqual(results, [])
        self.assertEqual(len(normalizer.unmapped), 1)

    def test_confidence_is_propagated(self) -> None:
        verified = self.normalizer.translate_raw(Vendor.NVIDIA, "79")
        self.assertEqual(verified[0].mapping_confidence, Confidence.VERIFIED)
        assumed = self.normalizer.translate_raw(Vendor.ASCEND, "HBM_ECC_UNCORRECTED")
        self.assertEqual(assumed[0].mapping_confidence, Confidence.ASSUMED)

    def test_target_keeps_gpu_scope(self) -> None:
        results = self.normalizer.translate_raw(
            Vendor.NVIDIA, "48", Target(node="gpu-node-15", gpu="1")
        )
        fault = results[0].fault
        self.assertEqual(fault.target.gpu, "1")


if __name__ == "__main__":
    unittest.main()
