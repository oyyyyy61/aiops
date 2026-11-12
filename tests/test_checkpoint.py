"""检查点存储测试：原子性、保留策略、完整性校验。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aiops.supervisor.checkpoint import CheckpointPolicy, CheckpointStore
from aiops.supervisor.supervisor import JobSpec, RecoveryRecord


class CheckpointStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.store = CheckpointStore(Path(self.tmp.name), "job-a", keep=3)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_save_and_latest(self) -> None:
        for step in (10, 20, 30):
            self.store.save_json(step, {"step": step})
        latest = self.store.latest()
        self.assertIsNotNone(latest)
        self.assertEqual(latest.step, 30)

    def test_keep_only_latest_n(self) -> None:
        for step in (10, 20, 30, 40, 50):
            self.store.save_json(step, {"step": step})
        steps = [m.step for m in self.store.list()]
        self.assertEqual(steps, [30, 40, 50])

    def test_no_tmp_directory_left(self) -> None:
        self.store.save_json(10, {"step": 10})
        leftovers = list(self.store.root.glob(".tmp-*"))
        self.assertEqual(leftovers, [])

    def test_verify_detects_corruption(self) -> None:
        meta = self.store.save_json(10, {"step": 10, "weights": [1, 2, 3]})
        self.assertTrue(self.store.verify(meta))
        state = Path(meta.path) / "state.json"
        state.write_text(json.dumps({"step": 10, "weights": [9]}), encoding="utf-8")
        self.assertFalse(self.store.verify(meta))

    def test_load_roundtrip(self) -> None:
        self.store.save_json(7, {"step": 7, "loss": 0.5})
        meta = self.store.latest()
        self.assertEqual(self.store.load_json(meta)["loss"], 0.5)

    def test_world_size_compatibility(self) -> None:
        self.store.save_json(10, {"step": 10}, world_size=4)
        self.assertIsNotNone(self.store.latest(compatible_world_size=4))
        self.assertIsNone(self.store.latest(compatible_world_size=2))

    def test_age(self) -> None:
        self.store.save_json(1, {"step": 1})
        age = self.store.age_s()
        self.assertIsNotNone(age)
        self.assertGreaterEqual(age, 0.0)


class CheckpointPolicyTest(unittest.TestCase):
    def test_interval_triggers(self) -> None:
        policy = CheckpointPolicy(interval_s=60, interval_steps=100)
        self.assertTrue(policy.due(0.0, 0, now=61.0, step=10))
        self.assertFalse(policy.due(0.0, 0, now=30.0, step=10))
        self.assertTrue(policy.due(0.0, 0, now=30.0, step=100))

    def test_rpo_budget(self) -> None:
        policy = CheckpointPolicy(rpo_budget_s=100)
        self.assertTrue(policy.rpo_exceeded(101))
        self.assertFalse(policy.rpo_exceeded(99))
        self.assertFalse(policy.rpo_exceeded(None))


class SupervisorModelTest(unittest.TestCase):
    def test_spawn_argv_rewrite(self) -> None:
        spec = JobSpec(
            name="j",
            argv=["python", "-m", "train", "--world-size", "2"],
            node="n1",
            world_size=2,
        )
        argv = spec.spawn_argv(resume=True, world_size=1, node="n2", exclude_ranks=["1"])
        self.assertIn("--resume", argv)
        self.assertEqual(argv[argv.index("--world-size") + 1], "1")
        self.assertEqual(argv[argv.index("--node") + 1], "n2")
        self.assertIn("--exclude-rank", argv)

    def test_rpo_with_and_without_emergency_checkpoint(self) -> None:
        record = RecoveryRecord(
            fault_code="UFC-GPU-001",
            fault_name="x",
            severity="S1",
            target="node=n1",
            strategy="evacuate",
            reason="r",
            failure_step=100,
            ckpt_step=100,
            rpo_steps=0,
            pre_fault_ckpt_step=80,
        )
        self.assertEqual(record.rpo_without_emergency, 20)
        self.assertEqual(record.rpo_steps, 0)


if __name__ == "__main__":
    unittest.main()
