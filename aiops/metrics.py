"""极简 Prometheus 文本指标导出。

不引入 prometheus_client：控制面往往要跑在裸机/容器里，少一个依赖就少一次
"装不上"。格式遵循 text exposition 0.0.4，可直接被 Prometheus 抓取或
`promtool check metrics` 校验。

导出的指标是 UFC 目录与告警规则的连接点：

    aiops_ufc_active{code="UFC-GPU-001",severity="S1"} 1
    aiops_recovery_total{strategy="evacuate",result="success"} 3
    aiops_recovery_downtime_seconds{strategy="evacuate"} 0.11
    aiops_recovery_rpo_steps{strategy="evacuate"} 0

对应 `python -m aiops catalog prometheus` 生成的规则：
``aiops_ufc_active{code="UFC-..."} == 1``。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Iterable, Mapping


def _escape(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _labels(pairs: Mapping[str, object]) -> str:
    parts = [f'{k}="{_escape(v)}"' for k, v in pairs.items() if v not in (None, "")]
    return "{" + ",".join(parts) + "}" if parts else ""


def render(
    detected: Iterable[Mapping],
    recoveries: Iterable[Mapping],
    active: Iterable[Mapping] = (),
) -> str:
    """渲染一个作业/场景的指标快照。

    ``detected`` 是本次运行检测到的全部故障（计数），``active`` 是此刻仍未关闭的
    故障（gauge）。两者分开是必要的：故障被处置后会 resolve，但"发生过"这个事实
    仍然要能用于统计。
    """

    detected = list(detected)
    recoveries = list(recoveries)
    lines: list[str] = [
        "# HELP aiops_ufc_detected_total 累计检测到的统一故障码",
        "# TYPE aiops_ufc_detected_total counter",
    ]
    counts: Counter[tuple[str, str]] = Counter()
    for fault in detected:
        counts[(str(fault.get("code")), str(fault.get("severity")))] += 1
    for (code, severity), count in sorted(counts.items()):
        lines.append(
            f"aiops_ufc_detected_total{_labels({'code': code, 'severity': severity})} {count}"
        )

    lines += [
        "# HELP aiops_ufc_active 当前未关闭的统一故障码",
        "# TYPE aiops_ufc_active gauge",
    ]
    active_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    for fault in active:
        active_counts[
            (str(fault.get("code")), str(fault.get("severity")), str(fault.get("target")))
        ] += 1
    for (code, severity, target), count in sorted(active_counts.items()):
        lines.append(
            f"aiops_ufc_active{_labels({'code': code, 'severity': severity, 'target': target})} {count}"
        )

    lines += [
        "# HELP aiops_recovery_total 恢复动作计数",
        "# TYPE aiops_recovery_total counter",
    ]
    total: Counter[tuple[str, str]] = Counter()
    downtime: dict[str, list[float]] = defaultdict(list)
    rpo: dict[str, list[float]] = defaultdict(list)
    for record in recoveries:
        strategy = str(record.get("strategy", "unknown"))
        result = "success" if record.get("verified") else "failure"
        total[(strategy, result)] += 1
        downtime[strategy].append(float(record.get("downtime_s", 0.0)))
        rpo[strategy].append(float(record.get("rpo_steps", 0)))
    for (strategy, result), count in sorted(total.items()):
        lines.append(f"aiops_recovery_total{_labels({'strategy': strategy, 'result': result})} {count}")

    lines += [
        "# HELP aiops_recovery_downtime_seconds 各策略最近一次停机时长",
        "# TYPE aiops_recovery_downtime_seconds gauge",
        "# HELP aiops_recovery_rpo_steps 各策略最近一次丢步数",
        "# TYPE aiops_recovery_rpo_steps gauge",
    ]
    for strategy, values in sorted(downtime.items()):
        lines.append(
            f"aiops_recovery_downtime_seconds{_labels({'strategy': strategy})} {values[-1]:.3f}"
        )
    for strategy, values in sorted(rpo.items()):
        lines.append(f"aiops_recovery_rpo_steps{_labels({'strategy': strategy})} {values[-1]:.0f}")
    return "\n".join(lines) + "\n"


def write(
    path,
    detected: Iterable[Mapping],
    recoveries: Iterable[Mapping],
    active: Iterable[Mapping] = (),
) -> None:
    from pathlib import Path

    Path(path).write_text(render(detected, recoveries, active), encoding="utf-8")
