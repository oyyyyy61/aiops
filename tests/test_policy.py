"""策略决策表测试：给定故障 + 上下文，必须选出预期策略。"""

from __future__ import annotations

import unittest

from aiops.detect.telemetry import QualityReport
from aiops.faultcode.models import Fault, Target
from aiops.faultcode.taxonomy import load_catalog
from aiops.policy.engine import JobContext, PolicyEngine


def make_fault(catalog, code_id: str, node: str = "n1", job: str = "j1") -> Fault:
    return Fault(
        code=catalog.get(code_id),
        target=Target(node=node, job=job),
        detected_at=100.0,
        onset_at=99.0,
    )


class PolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.catalog = load_catalog()
        cls.engine = PolicyEngine(cls.catalog)

    def ctx(self, **kwargs) -> JobContext:
        base = dict(job="j1", kind="train", node="n1", world_size=2)
        base.update(kwargs)
        return JobContext(**base)

    def test_s1_node_fatal_with_spare_evacuates(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-GPU-001"), self.ctx(spare_nodes=1))
        self.assertEqual(plan.strategy, "evacuate")

    def test_s1_without_spare_fails_fast(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-GPU-001"), self.ctx(spare_nodes=0))
        self.assertEqual(plan.strategy, "fail_fast")

    def test_uncorrectable_ecc_is_node_fatal(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-MEM-001"), self.ctx(spare_nodes=2))
        self.assertEqual(plan.strategy, "evacuate")
        self.assertIn("emergency_checkpoint", plan.actions)

    def test_job_hang_restarts_in_place(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-JOB-001"), self.ctx())
        self.assertEqual(plan.strategy, "restart_in_place")

    def test_job_hang_without_checkpoint_fails_fast(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-JOB-001"), self.ctx(checkpoint_available=False)
        )
        self.assertEqual(plan.strategy, "fail_fast")

    def test_gpu_suspect_migrates_when_spare(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-GPU-002"), self.ctx(spare_nodes=1))
        self.assertEqual(plan.strategy, "migrate")

    def test_gpu_suspect_restarts_without_spare(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-GPU-002"), self.ctx(spare_nodes=0))
        self.assertEqual(plan.strategy, "restart_in_place")

    def test_straggler_hardware_shrinks_without_spare(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-JOB-007"),
            self.ctx(spare_nodes=0, straggler_cause="hardware"),
        )
        self.assertEqual(plan.strategy, "shrink")

    def test_straggler_hardware_migrates_with_spare(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-JOB-007"),
            self.ctx(spare_nodes=1, straggler_cause="hardware"),
        )
        self.assertEqual(plan.strategy, "migrate")

    def test_straggler_load_does_not_move_gpu(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-JOB-007"),
            self.ctx(spare_nodes=1, straggler_cause="load"),
        )
        self.assertEqual(plan.strategy, "degrade")
        self.assertNotIn("migrate_job", plan.actions)

    def test_inference_faults_drain_and_restart(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-INF-001"), self.ctx(kind="inference")
        )
        self.assertEqual(plan.strategy, "drain_restart")
        self.assertEqual(plan.actions[0], "drain_instance")

    def test_inference_crash_loop_escalates(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-INF-004"), self.ctx(kind="inference")
        )
        self.assertEqual(plan.strategy, "fail_fast")

    def test_degraded_telemetry_forces_observe(self) -> None:
        plan = self.engine.decide(
            make_fault(self.catalog, "UFC-GPU-001"),
            self.ctx(
                spare_nodes=1,
                telemetry=QualityReport(state="STALE", score=0.3, reasons=["hb 陈旧"]),
            ),
        )
        self.assertEqual(plan.strategy, "observe")
        self.assertTrue(plan.degraded_by_telemetry)
        self.assertIn("STALE", plan.reason)

    def test_capacity_faults_degrade(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-MEM-004"), self.ctx())
        self.assertEqual(plan.strategy, "degrade")

    def test_s4_only_notifies(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-DRV-002"), self.ctx())
        self.assertEqual(plan.strategy, "observe")

    def test_plan_is_auditable(self) -> None:
        plan = self.engine.decide(make_fault(self.catalog, "UFC-GPU-001"), self.ctx(spare_nodes=1))
        payload = plan.as_dict()
        for key in ("strategy", "reason", "actions", "risk", "severity", "fault_code"):
            self.assertIn(key, payload)
        self.assertTrue(payload["reason"])


if __name__ == "__main__":
    unittest.main()
