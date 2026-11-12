"""慢节点识别。

慢节点有两种完全不同的根因，处置手段也相反：

* **硬件慢**：单卡频率/显存/互联退化，微基准跑不到同型号中位数；
* **负载慢**：数据分片不均、某个 rank 拿到长样本、CPU 侧 DataLoader 打满。

只用"步耗时"会把两者混为一谈，所以这里用两路证据：

1. 作业内步耗时（或吞吐）的稳健离群度：median + MAD，避免均值被离群点带偏；
2. 跨节点的微基准（all-reduce 带宽 / GEMM 有效算力）相对同型号基线的比值。

两者交叉后给出 ``hardware`` / ``load`` / ``unknown`` 判定，策略层据此决定
"迁移/隔离"还是"调数据/别动卡"。
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, Mapping

from ..faultcode.models import Fault, Severity, Target, Vendor
from ..faultcode.taxonomy import Catalog

MAD_TO_SIGMA = 0.6745


@dataclass
class StragglerReport:
    job: str
    window: int
    median_step_s: float
    slow_ranks: list[str]
    ratios: dict[str, float]
    score: float
    duration: int
    method: str = "step_time_mad"

    def as_dict(self) -> dict:
        return {
            "job": self.job,
            "window": self.window,
            "median_step_s": round(self.median_step_s, 4),
            "slow_ranks": self.slow_ranks,
            "ratios": {k: round(v, 3) for k, v in self.ratios.items()},
            "score": round(self.score, 3),
            "duration": self.duration,
            "method": self.method,
        }


@dataclass
class _Counter:
    streak: int = 0


class StepTimeMonitor:
    """按 rank 采集步耗时，输出离群的慢 rank。

    判定采用双条件：稳健 z 分数超过 ``z_threshold`` **且** 相对中位数超过
    ``ratio_threshold``（默认 1.35 倍）。后者防止在极稳的作业里把 3% 的抖动
    当成慢节点，前者防止在噪声大的作业里误判。
    """

    def __init__(
        self,
        job: str,
        window: int = 30,
        z_threshold: float = 3.0,
        ratio_threshold: float = 1.35,
        min_streak: int = 3,
        min_delta_s: float = 0.002,
    ):
        self.job = job
        self.window = window
        self.z_threshold = z_threshold
        self.ratio_threshold = ratio_threshold
        self.min_streak = min_streak
        # 除比值外还要有绝对增量：步耗时只有 1ms 时，1.35 倍也只是 0.35ms 的抖动，
        # 真实 2 卡实验里就因此把"掉卡瞬间"误判成慢节点。
        self.min_delta_s = min_delta_s
        self._samples: dict[str, list[float]] = {}
        self._streak: dict[str, _Counter] = {}

    def update(self, rank: str, step_time_s: float) -> None:
        if step_time_s <= 0:
            return
        buf = self._samples.setdefault(rank, [])
        buf.append(step_time_s)
        if len(buf) > self.window:
            del buf[: len(buf) - self.window]

    def update_many(self, step_times: Mapping[str, float]) -> None:
        for rank, value in step_times.items():
            self.update(rank, value)

    def _median_mad(self) -> tuple[float, float]:
        latest = [buf[-1] for buf in self._samples.values() if buf]
        if len(latest) < 2:
            return (latest[0] if latest else 0.0), 0.0
        median = statistics.median(latest)
        deviations = [abs(x - median) for x in latest]
        mad = statistics.median(deviations)
        return median, mad

    def evaluate(self) -> StragglerReport | None:
        ranks = [r for r, buf in self._samples.items() if buf]
        if len(ranks) < 2:
            return None
        median, mad = self._median_mad()
        if median <= 0:
            return None
        # MAD=0（所有 rank 完全一致）时用 1% 中位数兜底，避免除零与无穷大
        scale = mad * MAD_TO_SIGMA if mad > 0 else median * 0.01
        # MAD 的稳健 z 分数需要足够的同组样本：n=2 时 z 的理论上限只有 1.48，
        # 阈值 3.0 永远达不到。因此小规模作业只用比值 + 连续次数，不叠加 z。
        use_z = len(ranks) >= 5
        # 2 个 rank 时"中位数"就是两者的均值，value/median 的上限是 2.0，
        # 会把"一个 rank 慢 20 倍"压成 1.9。小规模用最小值做参照系。
        latest_values = [self._samples[r][-1] for r in ranks]
        reference = median if len(ranks) >= 3 else min(latest_values)
        ratios: dict[str, float] = {}
        slow: list[str] = []
        for rank in ranks:
            value = self._samples[rank][-1]
            ratio = value / reference if reference > 0 else 0.0
            z = (value - median) / scale if scale > 0 else 0.0
            ratios[rank] = ratio
            counter = self._streak.setdefault(rank, _Counter())
            above_ratio = ratio >= self.ratio_threshold and (value - reference) >= self.min_delta_s
            above_z = z >= self.z_threshold if use_z else True
            if above_ratio and above_z:
                counter.streak += 1
            else:
                counter.streak = 0
            if counter.streak >= self.min_streak:
                slow.append(rank)
        if not slow:
            return None
        score = max(ratios[r] for r in slow)
        duration = max(self._streak[r].streak for r in slow)
        return StragglerReport(
            job=self.job,
            window=self.window,
            median_step_s=median,
            slow_ranks=sorted(slow),
            ratios={r: ratios[r] for r in ranks},
            score=score,
            duration=duration,
        )

    def to_fault(self, report: StragglerReport, catalog: Catalog, node_of: Mapping[str, str] | None = None) -> Fault:
        """把慢节点报告转成 UFC-JOB-007，作用对象带上首个慢 rank 对应的节点。"""

        code = catalog.get("UFC-JOB-007")
        rank = report.slow_ranks[0]
        node = (node_of or {}).get(rank, "")
        return Fault(
            code=code,
            target=Target(job=report.job, node=node, gpu=rank if not node else ""),
            detected_at=0.0,
            onset_at=0.0,
            evidence=[f"step_skew={report.score:.2f}x median={report.median_step_s:.3f}s "
                      f"duration={report.duration} ranks={','.join(report.slow_ranks)}"],
        )


@dataclass
class BenchResult:
    """一次微基准结果。"""

    node: str
    gpu: str
    metric: str          # allreduce_gbps / gemm_tflops / h2d_gbps
    value: float
    ts: float = 0.0

    def as_dict(self) -> dict:
        return {"node": self.node, "gpu": self.gpu, "metric": self.metric, "value": self.value}


class BenchMonitor:
    """跨节点微基准：用同型号中位数做基线，低于阈值判为硬件慢。"""

    def __init__(self, fleet_baseline: Mapping[str, float] | None = None, ratio_threshold: float = 0.7):
        self.baseline = dict(fleet_baseline or {})
        self.ratio_threshold = ratio_threshold
        self.results: list[BenchResult] = []

    def record(self, result: BenchResult) -> float | None:
        self.results.append(result)
        base = self.baseline.get(result.metric)
        if not base:
            return None
        return result.value / base

    def baseline_from_fleet(self, metric: str) -> float:
        values = [r.value for r in self.results if r.metric == metric]
        if not values:
            return 0.0
        return statistics.median(values)

    def slow_devices(self, metric: str, window: int = 5) -> list[tuple[BenchResult, float]]:
        latest: dict[tuple[str, str], BenchResult] = {}
        for result in [r for r in self.results if r.metric == metric][-window * 64 :]:
            latest[(result.node, result.gpu)] = result
        base = self.baseline.get(metric) or self.baseline_from_fleet(metric)
        if base <= 0:
            return []
        out = []
        for result in latest.values():
            ratio = result.value / base
            if ratio < self.ratio_threshold:
                out.append((result, ratio))
        return sorted(out, key=lambda item: item[1])


def classify_root_cause(
    report: StragglerReport,
    bench: BenchMonitor | None,
    metric: str = "allreduce_gbps",
    node_of: Mapping[str, str] | None = None,
) -> str:
    """交叉判定：``hardware`` / ``load`` / ``unknown``。"""

    if bench is None:
        return "unknown"
    slow_nodes = set()
    for rank in report.slow_ranks:
        node = (node_of or {}).get(rank)
        if node:
            slow_nodes.add(node)
    if not slow_nodes:
        return "unknown"
    slow_devices = bench.slow_devices(metric)
    slow_node_names = {result.node for result, _ in slow_devices}
    if slow_nodes & slow_node_names:
        return "hardware"
    if slow_devices:
        return "load"
    return "load"
