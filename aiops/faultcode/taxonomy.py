"""UFC 目录的加载、校验与查询。

目录只做三件事：**定义**故障语义、**索引**厂商原始码、**约束**动作与验证项。
检测、决策、执行三层都从目录取数，避免"规则散落在告警 YAML 里"。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping

import yaml

from .models import (
    CHECKPOINT_POLICIES,
    KNOWN_ACTIONS,
    VERIFY_CHECKS,
    Confidence,
    FaultCode,
    Recurrence,
    Scope,
    Severity,
    SignalRule,
    Vendor,
    VendorMapping,
)

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
_UFC_RE = re.compile(r"^UFC-[A-Z]{3}-\d{3}$")


def _canon_code(raw: str) -> str:
    """把厂商原始码规范成可比较形式：``Xid 79`` / ``xid:79`` / ``79`` -> ``79``。"""

    text = str(raw).strip().lower()
    text = text.replace("xid", "").replace("error", "").strip()
    text = text.lstrip(":#- ").strip()
    text = re.sub(r"\s+", "", text)
    return text


@dataclass(frozen=True)
class Match:
    code: FaultCode
    mapping: VendorMapping
    via: str          # "code" | "pattern"
    matched: str      # 命中的原始码或正则


class Catalog:
    """内存中的故障码目录。"""

    def __init__(self, codes: Iterable[FaultCode], meta: Mapping[str, object] | None = None):
        self.codes: dict[str, FaultCode] = {c.id: c for c in codes}
        self.meta: dict = dict(meta or {})
        self._by_vendor_code: dict[tuple[Vendor, str], list[Match]] = {}
        self._by_vendor_pattern: dict[Vendor, list[tuple[re.Pattern[str], Match]]] = {}
        self._build_index()

    # ---------- 构造 ----------

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Catalog":
        p = Path(path) if path else CATALOG_PATH
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        codes = [cls._parse_code(item) for item in data.get("codes", [])]
        meta = {
            "version": data.get("version", "unknown"),
            "updated": data.get("updated", ""),
            "domains": data.get("domains", {}),
            "path": str(p),
        }
        return cls(codes, meta)

    @staticmethod
    def _parse_code(item: Mapping) -> FaultCode:
        signals = tuple(
            SignalRule(
                metric=s["metric"],
                op=s.get("op", "gt"),
                value=float(s.get("value", 0)),
                for_samples=int(s.get("for_samples", 1)),
                clear_samples=int(s.get("clear_samples", 2)),
                note=s.get("note", ""),
            )
            for s in item.get("signals", []) or []
        )
        vendor_map = tuple(
            VendorMapping(
                vendor=Vendor.parse(m["vendor"]),
                codes=tuple(str(c) for c in m.get("codes", []) or ()),
                patterns=tuple(m.get("patterns", []) or ()),
                confidence=Confidence(m.get("confidence", "assumed")),
                family=m.get("family", ""),
                note=m.get("note", ""),
            )
            for m in item.get("vendor_map", []) or []
        )
        rec = item.get("recurrence") or {}
        return FaultCode(
            id=item["id"],
            name=item["name"],
            domain=item["domain"],
            severity=Severity(item["severity"]),
            scope=Scope(item["scope"]),
            summary=item.get("summary", ""),
            actions=tuple(item.get("actions", []) or []),
            checkpoint=item.get("checkpoint", "periodic"),
            verify=tuple(item.get("verify", []) or ()),
            runbook=tuple(item.get("runbook", []) or ()),
            recurrence=Recurrence(
                max_retries=int(rec.get("max_retries", 1)),
                quarantine_after=int(rec.get("quarantine_after", 2)),
                backoff_s=float(rec.get("backoff_s", 5.0)),
            ),
            signals=signals,
            vendor_map=vendor_map,
            tags=tuple(item.get("tags", []) or ()),
            status=item.get("status", "active"),
        )

    def _build_index(self) -> None:
        for code in self.codes.values():
            for mapping in code.vendor_map:
                match = Match(code, mapping, "code", "")
                for raw in mapping.codes:
                    key = (mapping.vendor, _canon_code(raw))
                    self._by_vendor_code.setdefault(key, []).append(
                        Match(code, mapping, "code", raw)
                    )
                for pattern in mapping.patterns:
                    try:
                        compiled = re.compile(pattern, re.IGNORECASE)
                    except re.error as exc:  # 目录写错时立刻暴露
                        raise ValueError(f"{code.id} 的正则非法 {pattern!r}: {exc}") from exc
                    self._by_vendor_pattern.setdefault(mapping.vendor, []).append(
                        (compiled, Match(code, mapping, "pattern", pattern))
                    )

    # ---------- 查询 ----------

    def get(self, code_id: str) -> FaultCode:
        return self.codes[code_id]

    def __iter__(self) -> Iterator[FaultCode]:
        return iter(self.codes.values())

    def __len__(self) -> int:
        return len(self.codes)

    def by_domain(self, domain: str) -> list[FaultCode]:
        return sorted(
            (c for c in self.codes.values() if c.domain == domain.upper()),
            key=lambda c: c.id,
        )

    def by_severity(self, severity: Severity | str) -> list[FaultCode]:
        sev = Severity(severity) if isinstance(severity, str) else severity
        return sorted(
            (c for c in self.codes.values() if c.severity == sev), key=lambda c: c.id
        )

    def by_scope(self, scope: Scope) -> list[FaultCode]:
        return sorted((c for c in self.codes.values() if c.scope == scope), key=lambda c: c.id)

    def for_metric(self, metric: str) -> list[tuple[FaultCode, SignalRule]]:
        out: list[tuple[FaultCode, SignalRule]] = []
        for code in self.codes.values():
            for rule in code.signals:
                if rule.metric == metric:
                    out.append((code, rule))
        return out

    def vendor_codes(self, vendor: Vendor) -> dict[str, list[str]]:
        """列出某厂商的原始码到 UFC 的映射，用于生成对照表。"""

        table: dict[str, list[str]] = {}
        for code in self.codes.values():
            for mapping in code.vendor_map:
                if mapping.vendor != vendor:
                    continue
                for raw in mapping.codes:
                    table.setdefault(raw, []).append(code.id)
        return dict(sorted(table.items()))

    # ---------- 翻译：厂商原始信息 -> UFC ----------

    def match(self, vendor: Vendor, code: str = "", text: str = "") -> list[Match]:
        """按原始码优先、正则其次的顺序返回命中项（同一 UFC 只保留置信度最高的一条）。"""

        candidates: list[Match] = []
        # generic 表示跨厂商语义（如 NCCL/HCCL 超时、CUDA OOM、PCIe AER），对所有厂商生效
        for v in (vendor, Vendor.GENERIC) if vendor != Vendor.GENERIC else (Vendor.GENERIC,):
            if code:
                candidates.extend(self._by_vendor_code.get((v, _canon_code(code)), []))
            if text:
                for compiled, match in self._by_vendor_pattern.get(v, ()):
                    if compiled.search(text):
                        candidates.append(match)
        # 精确码命中是确定性的：一旦命中，就不再采纳文本正则推断，避免一行日志
        # 被翻译成多个语义（例如 Xid 74 同时被 "fatal" 误判成驱动崩溃）。
        if any(m.via == "code" for m in candidates):
            candidates = [m for m in candidates if m.via == "code"]
        # 去重：同一 UFC 保留置信度最高、且 code 匹配优先
        best: dict[str, Match] = {}
        for m in candidates:
            cur = best.get(m.code.id)
            score = (m.mapping.confidence.weight, 1 if m.via == "code" else 0)
            if cur is None:
                best[m.code.id] = m
                continue
            cur_score = (cur.mapping.confidence.weight, 1 if cur.via == "code" else 0)
            if score > cur_score:
                best[m.code.id] = m
        return sorted(best.values(), key=lambda m: (-m.mapping.confidence.weight, m.code.id))

    # ---------- 校验与导出 ----------

    def validate(self) -> list[str]:
        """返回问题列表；空列表表示通过。"""

        problems: list[str] = []
        seen: dict[str, str] = {}
        vendor_code_owner: dict[tuple[Vendor, str], str] = {}
        for code in self.codes.values():
            if not _UFC_RE.match(code.id):
                problems.append(f"{code.id}: 编号不符合 UFC-DOM-NNN")
            if code.id in seen:
                problems.append(f"{code.id}: 重复定义")
            seen[code.id] = code.name
            if code.domain not in self.meta.get("domains", {}):
                problems.append(f"{code.id}: 域 {code.domain} 未在 domains 中登记")
            if code.severity.auto_action and not code.verify:
                problems.append(f"{code.id}: {code.severity.value} 必须绑定 verify")
            for action in code.actions:
                if action not in KNOWN_ACTIONS:
                    problems.append(f"{code.id}: 未登记动作 {action}")
            for check in code.verify:
                if check not in VERIFY_CHECKS:
                    problems.append(f"{code.id}: 未登记验证项 {check}")
            if code.checkpoint not in CHECKPOINT_POLICIES:
                problems.append(f"{code.id}: checkpoint 策略非法 {code.checkpoint}")
            if code.severity == Severity.S1 and code.recurrence.max_retries > 1:
                problems.append(f"{code.id}: S1 故障重试次数过多（{code.recurrence.max_retries}）")
            if not code.vendor_map and not code.signals and code.severity.rank <= 2:
                problems.append(f"{code.id}: S1/S2 至少需要一条厂商映射或指标规则")
            for mapping in code.vendor_map:
                for raw in mapping.codes:
                    key = (mapping.vendor, _canon_code(raw))
                    owner = vendor_code_owner.get(key)
                    if owner and owner != code.id:
                        problems.append(
                            f"{code.id}: 厂商码 {mapping.vendor.value}:{raw} 已被 {owner} 占用"
                        )
                    else:
                        vendor_code_owner[key] = code.id
        return problems

    def stats(self) -> dict[str, object]:
        by_sev: dict[str, int] = {}
        by_vendor: dict[str, int] = {}
        for code in self.codes.values():
            by_sev[code.severity.value] = by_sev.get(code.severity.value, 0) + 1
            for vendor in code.vendor_labels:
                by_vendor[vendor] = by_vendor.get(vendor, 0) + 1
        return {
            "version": self.meta.get("version"),
            "total": len(self.codes),
            "by_severity": dict(sorted(by_sev.items())),
            "by_vendor_mapping": dict(sorted(by_vendor.items())),
            "domains": len(self.meta.get("domains", {})),
            "metric_rules": sum(len(c.signals) for c in self.codes.values()),
            "vendor_codes": sum(len(m.codes) for c in self.codes.values() for m in c.vendor_map),
            "patterns": sum(len(m.patterns) for c in self.codes.values() for m in c.vendor_map),
        }

    def to_markdown(self, domain: str | None = None) -> str:
        rows = ["| 故障码 | 名称 | 等级 | 范围 | 厂商映射 | 自动动作 |", "| --- | --- | --- | --- | --- | --- |"]
        codes = self.by_domain(domain) if domain else sorted(self.codes.values(), key=lambda c: c.id)
        for code in codes:
            vendors = "、".join(
                f"{m.vendor.value}({m.confidence.value})" for m in code.vendor_map
            ) or "指标规则"
            actions = "、".join(code.actions)
            rows.append(
                f"| `{code.id}` | {code.name} | {code.severity.value} | "
                f"{code.scope.value} | {vendors} | {actions} |"
            )
        return "\n".join(rows)

    def vendor_matrix_markdown(self) -> str:
        """生成「同义故障在不同厂商的原始码」对照表 —— 跨厂商标准化的核心交付物。"""

        groups: dict[str, list[FaultCode]] = {}
        for code in self.codes.values():
            if code.vendor_map:
                groups.setdefault(code.domain, []).append(code)
        lines = [
            "| 故障语义 | UFC | NVIDIA | AMD | 昇腾 | 置信度 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        for domain in sorted(groups):
            for code in sorted(groups[domain], key=lambda c: c.id):
                cells = {"nvidia": "—", "amd": "—", "ascend": "—"}
                for mapping in code.vendor_map:
                    raw = " / ".join(mapping.codes) or "—"
                    cells[mapping.vendor.value] = raw
                conf = "、".join(
                    f"{m.vendor.value}:{m.confidence.value}" for m in code.vendor_map
                )
                lines.append(
                    f"| {code.name} | `{code.id}` | {cells['nvidia']} | {cells['amd']} | "
                    f"{cells['ascend']} | {conf} |"
                )
        return "\n".join(lines)


def load_catalog(path: str | Path | None = None) -> Catalog:
    return Catalog.load(path)
