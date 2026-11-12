"""统一故障码目录的约束测试：格式、分级、动作与验证项。"""

from __future__ import annotations

import unittest

from aiops.faultcode.models import Severity, Vendor
from aiops.faultcode.taxonomy import load_catalog


class CatalogTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()

    def test_validation_passes(self) -> None:
        self.assertEqual(self.catalog.validate(), [])

    def test_catalog_has_cross_vendor_coverage(self) -> None:
        stats = self.catalog.stats()
        self.assertGreaterEqual(stats["total"], 40)
        vendors = stats["by_vendor_mapping"]
        self.assertIn("nvidia", vendors)
        self.assertIn("amd", vendors)
        self.assertIn("ascend", vendors)

    def test_s1_codes_bind_verification(self) -> None:
        for code in self.catalog.by_severity(Severity.S1):
            self.assertTrue(code.verify, f"{code.id} 缺少后验验证条件")
            self.assertLessEqual(code.recurrence.max_retries, 1, f"{code.id} S1 不应多次重试")

    def test_every_code_has_action_and_runbook(self) -> None:
        for code in self.catalog:
            self.assertTrue(code.actions, f"{code.id} 没有动作")
            self.assertGreaterEqual(len(code.runbook), 1, f"{code.id} 没有处理步骤")

    def test_vendor_code_unique(self) -> None:
        seen: dict[tuple[str, str], str] = {}
        for code in self.catalog:
            for mapping in code.vendor_map:
                for raw in mapping.codes:
                    key = (mapping.vendor.value, raw.strip().lower())
                    self.assertNotIn(key, seen, f"{code.id} 与 {seen.get(key)} 重复映射 {key}")
                    seen[key] = code.id

    def test_vendor_matrix_contains_known_rows(self) -> None:
        table = self.catalog.vendor_matrix_markdown()
        self.assertIn("UFC-MEM-001", table)
        self.assertIn("UFC-GPU-001", table)

    def test_confidence_is_declared(self) -> None:
        for code in self.catalog:
            for mapping in code.vendor_map:
                self.assertIn(mapping.confidence.value, {"verified", "community", "assumed"})

    def test_metric_rules_are_usable(self) -> None:
        rules = self.catalog.for_metric("gpu.ecc_uncorrected")
        self.assertTrue(rules)
        code, rule = rules[0]
        self.assertTrue(rule.evaluate(1.0))
        self.assertFalse(rule.evaluate(0.0))

    def test_translate_nvidia_dropped_card(self) -> None:
        matches = self.catalog.match(Vendor.NVIDIA, code="79")
        self.assertEqual([m.code.id for m in matches], ["UFC-GPU-001"])


if __name__ == "__main__":
    unittest.main()
