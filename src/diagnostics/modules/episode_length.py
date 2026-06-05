from __future__ import annotations

import numpy as np

from ..base import Diagnostic, DiagnosticContext
from ..io import load_episodes
from ..registry import register_diagnostic
from ..result import DiagnosticResult, Status


def _mean_episode_length(root) -> float:
    eps = load_episodes(root)
    lengths = [int(e["length"]) for e in eps if "length" in e]
    if not lengths:
        raise RuntimeError(f"no episode length entries under {root}")
    return float(np.mean(lengths))


@register_diagnostic(
    "EXP_03_Episode_Length_Inflation",
    category="temporal",
    thresholds={"critical": 2.0, "warning": 1.3},
)
class EpisodeLengthInflation(Diagnostic):
    """Direct temporal-inflation measurement from meta/episodes.jsonl lengths.

    Unlike EXP_02 (joint-space arc length), this metric reflects elapsed frame
    count at a presumed-constant control fps; it is the most literal proxy
    for "wallclock × fps" inflation visible in the dataset itself.
    """

    @classmethod
    def required_features(cls) -> set[str]:
        return set()  # reads meta/episodes.jsonl, not features

    def run(self, ctx: DiagnosticContext) -> DiagnosticResult:
        demo_len = _mean_episode_length(ctx.ref_root)
        cand_len = _mean_episode_length(ctx.cand_root)
        if demo_len <= 0.0:
            return DiagnosticResult(
                name=self.name, category=self.category, status=Status.ERROR,
                error="reference mean episode length is non-positive",
            )
        ratio = cand_len / demo_len

        ref_fps = ctx.ref_info.get("fps")
        cand_fps = ctx.cand_info.get("fps")
        metrics = {
            "demo_mean_length": round(demo_len, 3),
            "sft_mean_length": round(cand_len, 3),
            "length_ratio": round(ratio, 6),
        }
        if ref_fps and cand_fps and ref_fps == cand_fps:
            metrics["fps"] = float(ref_fps)
            metrics["demo_mean_seconds"] = round(demo_len / ref_fps, 3)
            metrics["sft_mean_seconds"] = round(cand_len / cand_fps, 3)

        crit = self.thresholds["critical"]
        warn = self.thresholds["warning"]
        if ratio > crit:
            status = Status.CRITICAL
        elif ratio > warn:
            status = Status.WARNING
        else:
            status = Status.OK

        narrative = [
            "Metric: mean episode frame count from meta/episodes.jsonl.",
            f"Thresholds: ratio > {crit} → CRITICAL; ratio > {warn} → WARNING.",
        ]
        if ref_fps != cand_fps:
            narrative.append(
                f"WARNING: fps mismatch (ref={ref_fps} vs cand={cand_fps}); "
                "length ratio is in frames, not seconds."
            )
        return DiagnosticResult(
            name=self.name, category=self.category, status=status,
            metrics=metrics, narrative=narrative,
        )

    @classmethod
    def report_template(cls) -> str:
        return (
            "## EXP_03 · Episode-Length Inflation\n\n"
            "### 1. Theoretical Hypothesis（猜想）\n"
            "若控制频率 fps 在两侧一致，episode 帧数即为完成任务所耗时长的直接代理。"
            "SFT 策略时长劣化在帧数比上应直接显现，不依赖任何运动学假设。\n\n"
            "### 2. Boundary Constraints & Prohibitions（边界与控制变量）\n"
            "- 控制变量：两侧 `meta/info.json.fps` 一致（若不一致则模块在 narrative 中显式声明，"
            "并仅汇报帧数比，不做秒数换算）；两侧均采用 frame-aligned 录制。\n"
            "- 边界：仅消费 `meta/episodes.jsonl` 中的 `length` 字段；不读 parquet。\n\n"
            "### 3. Experimental Protocol & Design（实验设计）\n"
            "- 对两侧分别取每 episode 的 `length` 字段，跨 episode 取均值，"
            "得到 `demo_mean_length` / `sft_mean_length`。\n"
            "- 报告 `length_ratio = sft_mean_length / demo_mean_length`。\n"
            "- 阈值：`ratio > {critical}` → CRITICAL；`ratio > {warning}` → WARNING；否则 OK。\n\n"
            "### 4. Quantitative Diagnostic Results & Causal Analysis（诊断结果与归因）\n"
            "- `demo_mean_length = {demo_mean_length}`，`sft_mean_length = {sft_mean_length}`，"
            "`length_ratio = {length_ratio}`，状态 **{status}**。\n"
            "- 归因：EXP_03 与 EXP_01/EXP_02 的相互关系是直接乘性印证——"
            "若 `length_ratio` 与 `(1/EXP_01.ratio) × EXP_02.path_len_ratio` 出现显著偏离，"
            "说明剩余时长来源（如推理延迟、动作平滑/分块）未被两者捕获。\n"
            "- 不引入额外数据；上述指标全部源自两侧 `meta/episodes.jsonl`。\n"
        )
