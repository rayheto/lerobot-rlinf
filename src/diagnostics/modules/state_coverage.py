from __future__ import annotations

import numpy as np

from ..base import Diagnostic, DiagnosticContext
from ..io import iter_episode_parquets, load_episode_parquet
from ..registry import register_diagnostic
from ..result import DiagnosticResult, Status
from ..schema import validate_pair

_FEATURE = "observation.state"
_MAX_REF = 8000
_MAX_QUERY = 4000
_BATCH = 512
_SEED = 0


def _collect_states(root, max_n: int, seed: int) -> np.ndarray:
    parts: list[np.ndarray] = []
    for p in iter_episode_parquets(root):
        cols = load_episode_parquet(p, columns=[_FEATURE])
        x = cols[_FEATURE]
        if x.ndim == 2 and x.shape[0] >= 1:
            parts.append(x.astype(np.float64))
    if not parts:
        raise RuntimeError(f"no usable {_FEATURE} under {root}")
    arr = np.concatenate(parts, axis=0)
    if arr.shape[0] > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(arr.shape[0], max_n, replace=False)
        arr = arr[idx]
    return arr


def _nn_min_distance(queries: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Return per-query L2 distance to the nearest anchor."""
    out = np.empty(queries.shape[0], dtype=np.float64)
    for i in range(0, queries.shape[0], _BATCH):
        q = queries[i : i + _BATCH]
        d2 = ((q[:, None, :] - anchors[None, :, :]) ** 2).sum(-1)
        out[i : i + _BATCH] = np.sqrt(d2.min(axis=1))
    return out


@register_diagnostic(
    "EXP_05_State_Coverage_Divergence",
    category="distributional",
    thresholds={"critical": 3.0, "warning": 2.0},
)
class StateCoverageDivergence(Diagnostic):
    """Measures how far candidate states sit from the reference state manifold.

    Baseline: mean nearest-neighbor distance from a held-out half of the
    reference set to its complement (intra-reference scale).
    Query:    mean nearest-neighbor distance from candidate states to the
    reference set.
    Ratio:    query_mean / baseline_mean — dimensionless OOD score.
    """

    @classmethod
    def required_features(cls) -> set[str]:
        return {_FEATURE}

    def run(self, ctx: DiagnosticContext) -> DiagnosticResult:
        mismatches = validate_pair(ctx.ref_info, ctx.cand_info, self.required_features())
        if mismatches:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.SKIPPED,
                error="schema mismatch: " + "; ".join(str(m) for m in mismatches),
            )

        ref = _collect_states(ctx.ref_root, _MAX_REF, _SEED)
        cand = _collect_states(ctx.cand_root, _MAX_QUERY, _SEED + 1)

        rng = np.random.default_rng(_SEED + 2)
        perm = rng.permutation(ref.shape[0])
        half = ref.shape[0] // 2
        ref_a, ref_b = ref[perm[:half]], ref[perm[half : 2 * half]]
        intra_d = _nn_min_distance(ref_a, ref_b)
        intra_mean = float(np.mean(intra_d))
        if intra_mean <= 0.0:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.ERROR,
                error="reference intra-NN distance is non-positive",
            )

        cand_d = _nn_min_distance(cand, ref)
        cand_mean = float(np.mean(cand_d))
        ratio = cand_mean / intra_mean

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
                "intra_ref_nn_mean": round(intra_mean, 6),
                "cand_to_ref_nn_mean": round(cand_mean, 6),
                "coverage_ratio": round(ratio, 6),
                "n_ref_sampled": int(ref.shape[0]),
                "n_cand_sampled": int(cand.shape[0]),
            },
            narrative=[
                f"Metric: mean 1-NN L2 distance from candidate `{_FEATURE}` to reference set, "
                "normalized by intra-reference half-vs-half 1-NN distance.",
                f"Subsampling: ref ≤ {_MAX_REF}, candidate ≤ {_MAX_QUERY} states.",
                f"Thresholds: ratio > {crit} → CRITICAL; ratio > {warn} → WARNING.",
            ],
        )

    @classmethod
    def report_template(cls) -> str:
        return (
            "## EXP_05 · State Coverage Divergence (OOD)\n\n"
            "### 1. Theoretical Hypothesis（猜想）\n"
            "若 SFT 策略在闭环执行中显著偏离示教 state 流形，候选 state 到示教 state 集合的"
            "最近邻距离将远大于示教集合内部的尺度。此为分布层面 OOD 的直接量化，独立于动作侧"
            "诊断（EXP_01/EXP_04）。\n\n"
            "### 2. Boundary Constraints & Prohibitions（边界与控制变量）\n"
            "- 控制变量：同一权重检查点、同一观察规约、同一 LeRobot v2.x dataset schema"
            "（`observation.state` dtype/shape 经 schema 校验通过）。\n"
            "- 边界：仅消费 `observation.state` 列；不做前向运动学、不调用模型；"
            "为控制计算成本，参考侧最多采样 {n_ref_sampled} 状态，候选侧最多 {n_cand_sampled}。\n\n"
            "### 3. Experimental Protocol & Design（实验设计）\n"
            "- 参考侧自身分半（`ref_a`、`ref_b`），计算 `ref_a` 到 `ref_b` 的逐点 1-NN L2 距离均值"
            "（`intra_ref_nn_mean`），作为同分布尺度基线。\n"
            "- 候选侧每个状态计算到参考侧全集的 1-NN L2 距离，取均值（`cand_to_ref_nn_mean`）。\n"
            "- 报告 `coverage_ratio = cand_to_ref_nn_mean / intra_ref_nn_mean`。\n"
            "- 阈值：`ratio > {critical}` → CRITICAL；`ratio > {warning}` → WARNING；否则 OK。\n\n"
            "### 4. Quantitative Diagnostic Results & Causal Analysis（诊断结果与归因）\n"
            "- `intra_ref_nn_mean = {intra_ref_nn_mean}`，"
            "`cand_to_ref_nn_mean = {cand_to_ref_nn_mean}`，"
            "`coverage_ratio = {coverage_ratio}`，状态 **{status}**。\n"
            "- 归因：高 `coverage_ratio` 直接对应 *OOD state visitation*，与 EXP_02（compounding "
            "error 的弧长后果）形成「原因侧 → 结果侧」两端联读。低 `coverage_ratio` 则反向支持："
            "策略仍处示教流形内，时长劣化更可能由动作侧因素主导（EXP_01/EXP_04）。\n"
            "- 不引入额外数据；上述指标全部源自两侧 dataset 既有 parquet。\n"
        )
