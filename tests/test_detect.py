"""检测层测试：去抖、事件生命周期、慢节点判定、观测质量门控。"""

from __future__ import annotations

import unittest

from aiops.collect.base import Sample, Snapshot
from aiops.detect.rules import Detector
from aiops.detect.straggler import BenchMonitor, BenchResult, StepTimeMonitor
from aiops.detect.telemetry import ActionGate, TelemetryMonitor
from aiops.faultcode.models import Target, Vendor, VendorEvent
from aiops.faultcode.taxonomy import load_catalog


def gpu_snapshot(ts: float, gpu_metrics: dict, node: str = "n1", gpu: str = "0") -> Snapshot:
    return Snapshot(
        ts=ts,
        samples=[
            Sample(
                ts=ts,
                kind="gpu",
                target=Target(node=node, gpu=gpu, vendor=Vendor.NVIDIA),
                metrics=gpu_metrics,
            )
        ],
    )


class RuleDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.catalog = load_catalog()

    def test_for_samples_debounce(self) -> None:
        detector = Detector(self.catalog)
        # UFC-MEM-001 的 gpu.ecc_uncorrected for_samples=1
        det = detector.observe(gpu_snapshot(1.0, {"gpu.ecc_uncorrected": 1}))
        self.assertEqual([f.code.id for f in det.new], ["UFC-MEM-001"])

    def test_slow_rule_needs_consecutive_samples(self) -> None:
        detector = Detector(self.catalog)
        # UFC-MEM-006: 显存占用 >97% 连续 6 次
        raised = []
        for i in range(5):
            det = detector.observe(gpu_snapshot(float(i), {"gpu.mem_used_ratio": 0.99}))
            raised.extend(f.code.id for f in det.new)
        self.assertEqual(raised, [])
        det = detector.observe(gpu_snapshot(6.0, {"gpu.mem_used_ratio": 0.99}))
        self.assertEqual([f.code.id for f in det.new], ["UFC-MEM-006"])

    def test_clear_requires_clear_samples(self) -> None:
        detector = Detector(self.catalog)
        detector.observe(gpu_snapshot(0.0, {"gpu.ecc_uncorrected": 1}))
        det = detector.observe(gpu_snapshot(1.0, {"gpu.ecc_uncorrected": 0}))
        self.assertEqual(det.resolved, [])
        det = detector.observe(gpu_snapshot(2.0, {"gpu.ecc_uncorrected": 0}))
        self.assertEqual([f.code.id for f in det.resolved], ["UFC-MEM-001"])

    def test_event_fault_expires_by_ttl(self) -> None:
        detector = Detector(self.catalog, event_ttl_s=10)
        snap = Snapshot(
            ts=0.0,
            events=[
                VendorEvent(
                    vendor=Vendor.NVIDIA,
                    code="79",
                    text="NVRM: Xid (PCI:0000:2d:00): 79, GPU has fallen off the bus.",
                    component="gpu-node-15",
                )
            ],
        )
        det = detector.observe(snap, now=0.0)
        self.assertEqual([f.code.id for f in det.new], ["UFC-GPU-001"])
        det = detector.observe(Snapshot(ts=30.0), now=30.0)
        self.assertEqual([f.code.id for f in det.resolved], ["UFC-GPU-001"])

    def test_duplicate_metric_does_not_re_raise(self) -> None:
        detector = Detector(self.catalog)
        detector.observe(gpu_snapshot(0.0, {"gpu.ecc_uncorrected": 1}))
        det = detector.observe(gpu_snapshot(1.0, {"gpu.ecc_uncorrected": 1}))
        self.assertEqual(det.new, [])
        self.assertEqual(len(det.updated), 1)


class StragglerTest(unittest.TestCase):
    def test_detects_slow_rank_with_three_ranks(self) -> None:
        monitor = StepTimeMonitor("job", min_streak=3)
        for _ in range(5):
            monitor.update_many({"0": 1.0, "1": 1.02, "2": 2.4})
            report = monitor.evaluate()
        self.assertIsNotNone(report)
        self.assertEqual(report.slow_ranks, ["2"])
        self.assertGreater(report.score, 2.0)

    def test_two_ranks_use_ratio_not_mad_z(self) -> None:
        monitor = StepTimeMonitor("job", min_streak=3)
        report = None
        for _ in range(4):
            monitor.update_many({"0": 1.0, "1": 3.0})
            report = monitor.evaluate()
        self.assertIsNotNone(report)
        self.assertEqual(report.slow_ranks, ["1"])

    def test_uniform_job_has_no_straggler(self) -> None:
        monitor = StepTimeMonitor("job", min_streak=3)
        for _ in range(6):
            monitor.update_many({"0": 1.0, "1": 1.01, "2": 0.99})
            report = monitor.evaluate()
        self.assertIsNone(report)

    def test_bench_classifies_hardware_vs_load(self) -> None:
        bench = BenchMonitor(fleet_baseline={"allreduce_gbps": 300})
        bench.record(BenchResult(node="n1", gpu="0", metric="allreduce_gbps", value=150))
        bench.record(BenchResult(node="n2", gpu="0", metric="allreduce_gbps", value=305))
        slow = bench.slow_devices("allreduce_gbps")
        self.assertEqual([r.node for r, _ in slow], ["n1"])


class TelemetryTest(unittest.TestCase):
    def test_fresh_source_is_ok(self) -> None:
        monitor = TelemetryMonitor(sources=["hb"], expected_interval_s=1.0)
        monitor.observe(Snapshot(ts=100.0, samples=[Sample(ts=100.0, kind="job", target=Target(job="j"), metrics={"job.step": 1})]))
        quality = monitor.evaluate(now=101.0)
        self.assertEqual(quality.state, "OK")
        self.assertTrue(quality.disruptive_allowed)

    def test_stale_source_blocks_disruptive_actions(self) -> None:
        monitor = TelemetryMonitor(sources=["hb"], expected_interval_s=1.0, stale_factor=3, blind_factor=8)
        monitor.observe(Snapshot(ts=100.0, samples=[]))
        gate = ActionGate(monitor)
        quality = monitor.evaluate(now=105.0)
        self.assertEqual(quality.state, "STALE")
        allowed, _ = gate.allowed("migrate_job", quality)
        self.assertFalse(allowed)
        allowed, _ = gate.allowed("emergency_checkpoint", quality)
        self.assertTrue(allowed)

    def test_blind_source_only_allows_safe_actions(self) -> None:
        monitor = TelemetryMonitor(sources=["hb"], expected_interval_s=1.0)
        monitor.observe(Snapshot(ts=100.0, samples=[]))
        quality = monitor.evaluate(now=200.0)
        self.assertEqual(quality.state, "BLIND")
        gate = ActionGate(monitor)
        self.assertFalse(gate.allowed("graceful_restart", quality)[0])
        self.assertTrue(gate.allowed("notify_only", quality)[0])


if __name__ == "__main__":
    unittest.main()
