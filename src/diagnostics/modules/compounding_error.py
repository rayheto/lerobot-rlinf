from __future__ import annotations

import numpy as np

from ..base import Diagnostic, DiagnosticContext
from ..io import iter_episode_parquets, load_episode_parquet
from ..registry import register_diagnostic
from ..result import DiagnosticResult, Status
from ..schema import validate_pair


def _mean_path_length(root, feature: str) -> float:
    """Mean over episodes of Σ_t ‖x_{t+1} − x_t‖_2 for the given feature column."""
    per_ep = []
    for p in iter_episode_parquets(root):
        cols = load_episode_parquet(p, columns=[feature])
        x = cols[feature]
        if x.ndim != 2 or x.shape[0] < 2:
            continue
        deltas = np.linalg.norm(np.diff(x, axis=0), axis=1)
        per_ep.append(float(deltas.sum()))
    if not per_ep:
        raise RuntimeError(f"no usable episodes under {root}")
    return float(np.mean(per_ep))


@register_diagnostic(
    "EXP_02_Compounding_Error",
    category="closed_loop",
    thresholds={"critical": 2.0, "warning": 1.5},
)
class CompoundingError(Diagnostic):
    FEATURE = "observation.state"

    @classmethod
    def required_features(cls) -> set[str]:
        return {cls.FEATURE}

    def run(self, ctx: DiagnosticContext) -> DiagnosticResult:
        mismatches = validate_pair(ctx.ref_info, ctx.cand_info, self.required_features())
        if mismatches:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.SKIPPED,
                error="schema mismatch: " + "; ".join(str(m) for m in mismatches),
            )

        demo_len = _mean_path_length(ctx.ref_root, self.FEATURE)
        cand_len = _mean_path_length(ctx.cand_root, self.FEATURE)
        if demo_len <= 0.0:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.ERROR,
                error="reference path length is non-positive",
            )
        ratio = cand_len / demo_len

        crit = self.thresholds["critical"]
        warn = self.thresholds["warning"]
        if ratio > crit:
            status = Status.CRITICAL
        elif ratio > warn:
            status = Status.WARNING
        else:
            status = Status.OK

        return DiagnosticResult(
            name=self.name, category=self.category, status=status,
            metrics={
                "demo_path_len": round(demo_len, 6),
                "sft_path_len": round(cand_len, 6),
                "path_len_ratio": round(ratio, 6),
            },
            narrative=[
                f"Metric: per-episode cumulative L2 arc length over `{self.FEATURE}`, "
                "aggregated by mean over episodes.",
                f"Thresholds: ratio > {crit} → CRITICAL; ratio > {warn} → WARNING.",
            ],
        )

    @classmethod
    def report_template(cls) -> str:
        return (
            "## EXP_02 · Compounding Error in Closed-Loop Rollout\n\n"
            "### 1. Theoretical Hypothesis（猜想）\n"
            "行为克隆策略在分布外状态上的小幅误差经闭环执行累积"
            "（compounding error；Ross & Bagnell, 2010），表现为 OOD trajectory divergence"
            "与持续修正动作链，宏观上即关节空间累计弧长（path length）膨胀。\n\n"
            "### 2. Boundary Constraints & Prohibitions（边界与控制变量）\n"
            "- 控制变量：同一权重检查点、同一归一化统计、同一观察规约、同一任务定义、"
            "同一 LeRobot v2.x dataset schema（`observation.state` dtype/shape 经 schema 校验通过）。\n"
            "- 边界：路径长度严格定义为 `Σ_t ‖x_{{t+1}} − x_t‖₂`，其中 `x_t` 取 `observation.state`；"
            "不做前向运动学、不引入末端坐标系假设、不重新归一化。\n\n"
            "### 3. Experimental Protocol & Design（实验设计）\n"
            "- 对参考侧与待测侧的每一条 episode parquet，按 `frame_index` 顺序取 "
            "`observation.state` 时间序列，逐帧差分取 L2，累加得到该 episode 的关节空间弧长。\n"
            "- 跨 episode 取均值作为聚合量（`demo_path_len` / `sft_path_len`），"
            "报告比值 `path_len_ratio = sft_path_len / demo_path_len`。\n"
            "- 阈值：`ratio > {critical}` → CRITICAL；`ratio > {warning}` → WARNING；否则 OK。\n\n"
            "### 4. Quantitative Diagnostic Results & Causal Analysis（诊断结果与归因）\n"
            "- `demo_path_len = {demo_path_len}`，`sft_path_len = {sft_path_len}`，"
            "`path_len_ratio = {path_len_ratio}`，状态 **{status}**。\n"
            "- 归因链：*OOD state visitation → policy uncertainty under L2 BC objective → "
            "corrective sub-trajectories → cumulative arc-length inflation*。\n"
            "- 不引入额外数据；上述指标全部源自两侧 dataset 既有 parquet。\n"
        )
