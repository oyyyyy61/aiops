"""命令行入口：``python -m aiops <子命令>``。

面向三类使用者：

* 值班同学：``translate`` / ``diagnose`` 查一条日志或一个厂商码到底算什么故障；
* 平台同学：``catalog stats|matrix|prometheus`` 生成规范表格与告警规则；
* 算法同学：``demo`` 跑故障场景回放，``report`` 看汇总。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .faultcode.models import Severity, Target, Vendor
from .faultcode.normalizer import Normalizer
from .faultcode.taxonomy import load_catalog


def _catalog_stats(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(json.dumps(catalog.stats(), ensure_ascii=False, indent=2))
    problems = catalog.validate()
    if problems:
        print(f"\n[!] 目录校验发现 {len(problems)} 个问题：", file=sys.stderr)
        for problem in problems:
            print(f"  - {problem}", file=sys.stderr)
        return 1
    print("\n目录校验：通过", file=sys.stderr)
    return 0


def _catalog_matrix(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    print(catalog.vendor_matrix_markdown() if args.vendor_matrix else catalog.to_markdown(args.domain))
    return 0


def _translate(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    normalizer = Normalizer(catalog)
    target = Target(node=args.node, gpu=args.gpu, job=args.job, vendor=Vendor.parse(args.vendor))
    results = normalizer.translate_raw(Vendor.parse(args.vendor), args.code, target)
    if args.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
        return 0 if results else 2
    if not results:
        print(f"未映射：{args.vendor}:{args.code}（需要新增 UFC 码或厂商映射）", file=sys.stderr)
        return 2
    for r in results:
        code = r.fault.code
        print(f"{r.vendor.value}:{args.code}  ->  {code.id} {code.name}")
        print(f"  等级 {code.severity.value} / 影响范围 {code.scope.value} / 映射置信度 {r.mapping_confidence.value}")
        print(f"  摘要：{code.summary}")
        print(f"  自动动作：{', '.join(code.actions)}")
        print(f"  验证条件：{', '.join(code.verify) or '仅告警'}")
        for step in code.runbook:
            print(f"  - {step}")
    return 0


def _diagnose(args: argparse.Namespace) -> int:
    catalog = load_catalog(args.catalog)
    normalizer = Normalizer(catalog)
    lines = Path(args.file).read_text(encoding="utf-8", errors="replace").splitlines()
    results = normalizer.normalize_lines(
        Vendor.parse(args.vendor), lines, Target(node=args.node, vendor=Vendor.parse(args.vendor))
    )
    if args.json:
        print(json.dumps([r.as_dict() for r in results], ensure_ascii=False, indent=2))
        return 0 if results else 2
    if not results:
        print("没有识别出已知故障码", file=sys.stderr)
        return 2
    for r in results:
        code = r.fault.code
        print(f"[{code.severity.value}] {code.id} {code.name}  <- {r.vendor.value}:{r.raw[:60]!r}")
    codes = sorted({r.fault.code.id for r in results})
    print(f"\n共 {len(results)} 条事件，映射到 {len(codes)} 个 UFC：{', '.join(codes)}")
    return 0


def _collect(args: argparse.Namespace) -> int:
    vendor = Vendor.parse(args.vendor)
    if vendor == Vendor.NVIDIA:
        from .collect.nvidia import NvidiaCollector

        collector = NvidiaCollector(args.node, args.source)
    elif vendor == Vendor.AMD:
        from .collect.amd import AmdCollector

        collector = AmdCollector(args.node, args.source)
    elif vendor == Vendor.ASCEND:
        from .collect.ascend import AscendCollector

        collector = AscendCollector(args.node, args.source)
    else:
        print("不支持的厂商", file=sys.stderr)
        return 2
    snap = collector.collect()
    print(json.dumps(snap.as_dict(), ensure_ascii=False, indent=2))
    return 0 if snap.source_up else 3


def _catalog_prometheus(args: argparse.Namespace) -> int:
    """把 UFC 目录导出成 PrometheusRule，告警直接带上统一故障码标签。"""

    catalog = load_catalog(args.catalog)
    lines = [
        "apiVersion: monitoring.coreos.com/v1",
        "kind: PrometheusRule",
        "metadata:",
        "  name: aiops-unified-fault-codes",
        "  labels:",
        "    category: aiops",
        "spec:",
        "  groups:",
        "    - name: aiops.ufc",
        "      rules:",
    ]
    for code in sorted(catalog, key=lambda c: (c.severity.rank, c.id)):
        if code.severity.rank > 2:
            continue
        lines += [
            f"        - alert: {code.id}",
            "          expr: aiops_ufc_active{" + f'code="{code.id}"' + "} == 1",
            "          labels:",
            f"            severity: {code.severity.value.lower()}",
            "            category: aiops",
            f"            ufc: {code.id}",
            f"            scope: {code.scope.value}",
            "          annotations:",
            f"            summary: \"{code.name}\"",
            f"            description: \"{code.summary}\"",
            f"            runbook: \"{code.runbook[0] if code.runbook else '见 UFC 目录'}\"",
        ]
    print("\n".join(lines))
    return 0


def _demo(args: argparse.Namespace) -> int:
    from demo.runner import run_all

    paths = []
    for pattern in args.scenarios:
        paths.extend(sorted(Path().glob(pattern)) if any(ch in pattern for ch in "*?[") else [Path(pattern)])
    if not paths:
        print("没有找到场景文件", file=sys.stderr)
        return 2
    outcomes = run_all(paths, out_root=args.out)
    print(f"\n报告：{Path(args.out) / 'report.md'}")
    return 0 if all(o.expect_ok for o in outcomes) else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="aiops", description="AI 集群自动运维探索原型")
    parser.add_argument("--catalog", default=None, help="UFC 目录路径（默认内置 catalog.yaml）")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("catalog", help="目录查询与导出")
    p.add_argument("action", choices=["stats", "matrix", "prometheus"])
    p.add_argument("--domain", default=None, help="matrix 时只看某个域，如 GPU")
    p.add_argument("--vendor-matrix", action="store_true", help="输出跨厂商对照表")
    p.set_defaults(func=_catalog_stats)

    p = sub.add_parser("translate", help="把厂商原始码翻译成 UFC")
    p.add_argument("--vendor", required=True)
    p.add_argument("--code", required=True)
    p.add_argument("--node", default="")
    p.add_argument("--gpu", default="")
    p.add_argument("--job", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_translate)

    p = sub.add_parser("diagnose", help="诊断一段厂商日志")
    p.add_argument("--vendor", required=True)
    p.add_argument("--file", required=True)
    p.add_argument("--node", default="")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=_diagnose)

    p = sub.add_parser("collect", help="采集一次硬件遥测并输出统一指标")
    p.add_argument("--vendor", required=True)
    p.add_argument("--source", default=None, help="XML/JSON/文本文件；省略则调用厂商工具")
    p.add_argument("--node", default="localhost")
    p.set_defaults(func=_collect)

    p = sub.add_parser("demo", help="跑故障场景回放实验")
    p.add_argument("scenarios", nargs="*", default=["demo/scenarios/*.yaml"])
    p.add_argument("--out", default="results")
    p.set_defaults(func=_demo)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "catalog":
        args.func = {
            "stats": _catalog_stats,
            "matrix": _catalog_matrix,
            "prometheus": _catalog_prometheus,
        }[args.action]
        if args.action == "matrix" and args.vendor_matrix:
            pass
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
