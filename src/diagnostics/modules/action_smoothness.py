from __future__ import annotations

import numpy as np

from ..base import Diagnostic, DiagnosticContext
from ..io import iter_episode_parquets, load_episode_parquet
from ..registry import register_diagnostic
from ..result import DiagnosticResult, Status
from ..schema import validate_pair


def _mean_step_delta(root) -> float:
    """Mean over episodes of mean(‖a_{t+1} − a_t‖_2)."""
    per_ep = []
    for p in iter_episode_parquets(root):
        cols = load_episode_parquet(p, columns=["action"])
        a = cols["action"]
        if a.ndim != 2 or a.shape[0] < 2:
            continue
        d = np.linalg.norm(np.diff(a, axis=0), axis=1)
        per_ep.append(float(d.mean()))
    if not per_ep:
        raise RuntimeError(f"no usable episodes under {root}")
    return float(np.mean(per_ep))


@register_diagnostic(
    "EXP_04_Action_Smoothness",
    category="temporal",
    thresholds={
        "critical_low": 0.4, "warning_low": 0.7,
        "warning_high": 1.5, "critical_high": 2.5,
    },
)
class ActionSmoothness(Diagnostic):
    """Per-step action delta magnitude (jerk proxy).

    Bidirectional thresholds:
    - ratio < critical_low / warning_low → stair-step / chunk-hold artifact
      (e.g. action-chunking + intra-chunk constant hold, EMA over-smoothing).
    - ratio > warning_high / critical_high → high-frequency jitter
      (e.g. policy variance / numeric instability).
    """

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

        demo_d = _mean_step_delta(ctx.ref_root)
        cand_d = _mean_step_delta(ctx.cand_root)
        if demo_d <= 0.0:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.ERROR,
                error="reference step-delta is non-positive",
            )
        ratio = cand_d / demo_d

        t = self.thresholds
        if ratio < t["critical_low"] or ratio > t["critical_high"]:
            status = Status.CRITICAL
        elif ratio < t["warning_low"] or ratio > t["warning_high"]:
            status = Status.WARNING
        else:
            status = Status.OK

        if ratio < 1.0:
            direction = "stair-step / chunk-hold suppression"
        else:
            direction = "high-frequency jitter / variance inflation"

        return DiagnosticResult(
            name=self.name, category=self.category, status=status,
            metrics={
                "demo_mean_step_delta": round(demo_d, 6),
                "sft_mean_step_delta": round(cand_d, 6),
                "smoothness_ratio": round(ratio, 6),
            },
            narrative=[
                "Metric: per-episode mean ‖a_{t+1} − a_t‖_2, aggregated by mean over episodes.",
                f"Direction: {direction}.",
                f"Thresholds: ratio ∉ [{t['warning_low']}, {t['warning_high']}] → WARNING; "
                f"ratio ∉ [{t['critical_low']}, {t['critical_high']}] → CRITICAL.",
            ],
        )

    @classmethod
    def report_template(cls) -> str:
        return (
            "## EXP_04 · Per-Step Action Smoothness\n\n"
            "### 1. Theoretical Hypothesis（猜想）\n"
            "动作分块（action chunking）与块内常值保持、或推理外加 EMA 平滑，会显著压低"
            "相邻帧动作差分的 L2 范数（stair-step artifact）；反之策略方差膨胀会抬高该量"
            "（high-frequency jitter）。二者均偏离示教侧自然的 actuation 节律，且与时长劣化"
            "因果可分离。\n\n"
            "### 2. Boundary Constraints & Prohibitions（边界与控制变量）\n"
            "- 控制变量：同一权重检查点、同一归一化统计、同一观察规约、同一 LeRobot v2.x "
            "dataset schema（action dtype/shape 经 schema 校验通过）。\n"
            "- 边界：仅消费 `action` 列；不调用模型；不重新归一化。\n\n"
            "### 3. Experimental Protocol & Design（实验设计）\n"
            "- 对两侧每 episode 计算 `mean_t ‖a_{{t+1}} − a_t‖_2`，跨 episode 取均值，"
            "得到 `demo_mean_step_delta` / `sft_mean_step_delta`。\n"
            "- 报告 `smoothness_ratio = sft_mean_step_delta / demo_mean_step_delta`。\n"
            "- 阈值（双向）：偏离 [{warning_low}, {warning_high}] → WARNING；"
            "偏离 [{critical_low}, {critical_high}] → CRITICAL。\n\n"
            "### 4. Quantitative Diagnostic Results & Causal Analysis（诊断结果与归因）\n"
            "- `demo_mean_step_delta = {demo_mean_step_delta}`，"
            "`sft_mean_step_delta = {sft_mean_step_delta}`，"
            "`smoothness_ratio = {smoothness_ratio}`，状态 **{status}**。\n"
            "- 归因：`smoothness_ratio < 1` 印证 chunking / EMA 压平；"
            "`smoothness_ratio > 1` 印证策略方差膨胀。两种情形对时长劣化的贡献方向相反，"
            "需与 EXP_01（动作散度）联读以消歧：散度收缩 + 平滑度收缩 → chunking 主因；"
            "散度收缩 + 平滑度膨胀 → 策略 mode covering 同时叠加高频噪声修正链。\n"
            "- 不引入额外数据；上述指标全部源自两侧 dataset 既有 parquet。\n"
        )
