from __future__ import annotations

import numpy as np

from ..base import Diagnostic, DiagnosticContext
from ..io import (
    iter_episode_parquets,
    load_episode_parquet,
    load_episodes_stats,
)
from ..registry import register_diagnostic
from ..result import DiagnosticResult, Status
from ..schema import validate_pair


def _mean_action_dispersion(root, stats: list[dict] | None) -> float:
    """Mean over episodes of ‖std(action_t)‖_2 — a proxy for action dispersion.

    Fast path consumes precomputed episodes_stats; slow path streams parquets.
    """
    if stats is not None:
        per_ep = []
        for ep in stats:
            std = ep.get("stats", {}).get("action", {}).get("std")
            if std is None:
                continue
            per_ep.append(float(np.linalg.norm(np.asarray(std, dtype=np.float64))))
        if per_ep:
            return float(np.mean(per_ep))

    per_ep = []
    for p in iter_episode_parquets(root):
        cols = load_episode_parquet(p, columns=["action"])
        a = cols["action"]
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        per_ep.append(float(np.linalg.norm(np.std(a, axis=0))))
    if not per_ep:
        raise RuntimeError(f"no usable episodes under {root}")
    return float(np.mean(per_ep))


@register_diagnostic(
    "EXP_01_Mode_Averaging",
    category="distributional",
    thresholds={"critical": 0.5, "warning": 0.8},
)
class ModeAveraging(Diagnostic):
    @classmethod
    def required_features(cls) -> set[str]:
        return {"action"}

    def run(self, ctx: DiagnosticContext) -> DiagnosticResult:
        mismatches = validate_pair(ctx.ref_info, ctx.cand_info, self.required_features())
        if mismatches:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.SKIPPED,
                error="schema mismatch: " + "; ".join(str(m) for m in mismatches),
            )

        ref_stats = load_episodes_stats(ctx.ref_root)
        cand_stats = load_episodes_stats(ctx.cand_root)
        demo_mean = _mean_action_dispersion(ctx.ref_root, ref_stats)
        sft_mean = _mean_action_dispersion(ctx.cand_root, cand_stats)
        if demo_mean <= 0.0:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.ERROR,
                error="reference dispersion is non-positive",
            )
        ratio = sft_mean / demo_mean

        crit = self.thresholds["critical"]
        warn = self.thresholds["warning"]
        if ratio < crit:
            status = Status.CRITICAL
        elif ratio < warn:
            status = Status.WARNING
        else:
            status = Status.OK

        return DiagnosticResult(
            name=self.name, category=self.category, status=status,
            metrics={
                "demo_mean": round(demo_mean, 6),
                "sft_mean": round(sft_mean, 6),
                "ratio": round(ratio, 6),
            },
            narrative=[
                "Metric: per-episode ‖std(action_t)‖_2, aggregated by mean over episodes.",
                "Fast path uses meta/episodes_stats.jsonl when available; falls back to streaming parquets.",
                f"Thresholds: ratio < {crit} → CRITICAL; ratio < {warn} → WARNING.",
            ],
        )

    @classmethod
    def report_template(cls) -> str:
        return (
            "## EXP_01 · Mode-Covering Induced Action Magnitude Collapse\n\n"
            "### 1. Theoretical Hypothesis（猜想）\n"
            "在 L2 行为克隆目标下，策略对多峰示教动作分布执行 *mode covering*，"
            "其条件均值输出导致动作分布散度（per-episode action std 的 L2 范数）"
            "相对参考侧显著收缩，直接对应执行轨迹的低致动速率与时长膨胀。\n\n"
            "### 2. Boundary Constraints & Prohibitions（边界与控制变量）\n"
            "- 控制变量：同一权重检查点、同一归一化统计、同一观察规约、同一任务定义、"
            "同一评测协议、同一 LeRobot v2.x dataset schema（action dtype/shape 经 schema 校验通过）。\n"
            "- 边界：仅消费 `meta/episodes_stats.jsonl` 中既有统计或逐 parquet 重算；"
            "不引入新样本、不调用模型推理、不重新归一化。\n\n"
            "### 3. Experimental Protocol & Design（实验设计）\n"
            "- 对参考侧与待测侧分别计算每 episode `action_t` 的逐维标准差向量，"
            "取其 L2 范数作为该 episode 的动作散度；对所有 episode 取均值作为聚合量"
            "（`demo_mean` / `sft_mean`）。\n"
            "- 报告比值 `ratio = sft_mean / demo_mean`。\n"
            "- 阈值：`ratio < {critical}` → CRITICAL；`ratio < {warning}` → WARNING；否则 OK。\n\n"
            "### 4. Quantitative Diagnostic Results & Causal Analysis（诊断结果与归因）\n"
            "- `demo_mean = {demo_mean}`，`sft_mean = {sft_mean}`，`ratio = {ratio}`，"
            "状态 **{status}**。\n"
            "- 归因链：*mode covering → action dispersion collapse → reduced per-step "
            "actuation magnitude → temporal inflation*。\n"
            "- 不引入额外数据；上述指标全部源自两侧 dataset 既有内容。\n"
        )
