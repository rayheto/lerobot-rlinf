from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .registry import REGISTRY
from .result import DiagnosticResult, Status

FORBIDDEN = re.compile(r"犹豫|迷糊|hesitate|confused", re.IGNORECASE)


def _abstract(meta: dict[str, Any] | None) -> str:
    if not meta:
        return (
            "## Abstract\n\n"
            "[abstract metadata required — pass `--meta-json` with keys: "
            "`task`, `num_demos`, `success_rate_range`, `temporal_inflation_range`]\n"
        )
    keys = ("task", "num_demos", "success_rate_range", "temporal_inflation_range")
    missing = [k for k in keys if k not in meta]
    if missing:
        return (
            "## Abstract\n\n"
            f"[abstract metadata incomplete; missing: {missing}]\n"
        )
    return (
        "## Abstract\n\n"
        f"针对 `{meta['task']}` 任务（{meta['num_demos']} 条专家示教，"
        f"LeRobot v2.x schema），SFT 后策略呈现成功率 {meta['success_rate_range']} 但"
        f"执行时长劣化 {meta['temporal_inflation_range']} 的失败模式。本报告基于一个"
        "解耦的、仅消费两个同 schema 数据集的诊断框架，量化两项失效成因："
        "动作分布散度塌缩（mode covering）与关节空间累计弧长膨胀（compounding error）。\n"
    )


def _format_template(template: str, result: DiagnosticResult) -> str:
    fields: dict[str, Any] = {"status": result.status.value, **result.metrics}
    cls = REGISTRY.get(result.name)
    if cls is not None:
        for k, v in cls.thresholds.items():
            fields[k] = v
    try:
        return template.format(**fields)
    except KeyError as e:
        return template + f"\n\n[template render error: missing field {e}]\n"


def _render_module(result: DiagnosticResult) -> str:
    if result.status == Status.ERROR:
        return (
            f"## {result.name}\n\n"
            f"状态 **{result.status.value}**。\n\n"
            f"```\n{result.error or 'no detail'}\n```\n"
        )
    if result.status == Status.SKIPPED:
        return (
            f"## {result.name}\n\n"
            f"状态 **SKIPPED**：{result.error or 'schema mismatch'}。本模块未产出指标。\n"
        )
    cls = REGISTRY.get(result.name)
    template = cls.report_template() if cls is not None else ""
    if not template:
        body = "\n".join(f"- `{k} = {v}`" for k, v in result.metrics.items())
        return f"## {result.name}\n\n状态 **{result.status.value}**。\n\n{body}\n"
    return _format_template(template, result)


def _discussion(results: list[DiagnosticResult]) -> str:
    by_name = {r.name: r for r in results}
    r1 = by_name.get("EXP_01_Mode_Averaging")
    r2 = by_name.get("EXP_02_Compounding_Error")
    if r1 is None or r2 is None:
        return ""
    if r1.status in (Status.ERROR, Status.SKIPPED) or r2.status in (Status.ERROR, Status.SKIPPED):
        return ""
    ratio = r1.metrics.get("ratio")
    path_ratio = r2.metrics.get("path_len_ratio")
    if ratio is None or path_ratio is None or ratio <= 0:
        return ""
    upper = (1.0 / ratio) * path_ratio
    return (
        "## Discussion\n\n"
        "两项指标在因果上彼此独立但在时长后果上乘性叠加：动作分布散度塌缩"
        "（EXP_01）压低每步致动幅度，关节空间弧长膨胀（EXP_02）加长闭环执行路径。"
        f"定性印证：`T_sft / T_demo ≈ (1/ratio) × path_len_ratio = "
        f"{(1.0/ratio):.3f} × {path_ratio:.3f} ≈ {upper:.3f}×`。"
        "该乘积仅作为量级印证，不构成严格证明；它未控制控制频率、动作平滑滤波、"
        "动力学响应等额外混淆因素。\n"
    )


def render(results: list[DiagnosticResult], meta: dict[str, Any] | None = None) -> str:
    parts = [
        "# Diagnostic Evaluation of SFT-Induced Temporal Inflation\n",
        _abstract(meta),
    ]
    for r in results:
        parts.append(_render_module(r))
    disc = _discussion(results)
    if disc:
        parts.append(disc)
    text = "\n".join(parts).rstrip() + "\n"
    hit = FORBIDDEN.search(text)
    if hit:
        raise RuntimeError(f"report contains forbidden colloquial term: {hit.group(0)!r}")
    return text


def render_to_files(
    results: list[DiagnosticResult],
    out_json: Path | None,
    out_md: Path | None,
    meta: dict[str, Any] | None = None,
) -> None:
    if out_json is not None:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        payload = {"results": [r.to_dict() for r in results], "meta": meta or {}}
        out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if out_md is not None:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(render(results, meta))
