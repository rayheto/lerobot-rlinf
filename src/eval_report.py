"""Bar chart (success rate + FAST rate, with error bars) from eval.py --watch output.

Reads <exp_dir>/eval_summary.jsonl (one line per evaluated checkpoint, written by
src/eval.py --watch) and renders a two-panel bar chart: success_rate +/- std, and
fast_rate, one bar per checkpoint step.

Example:
    python src/eval_report.py --exp-name=so101_pick_orange_lora_v0 \
        --title="SFT eval" --out=/tmp/eval_report.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-name")
    p.add_argument("--config-name", default="pi05_lora_so101_pick_orange")
    p.add_argument("--checkpoint-base-dir", default=str(REPO_ROOT / "outputs"))
    p.add_argument("--out", default=None, help="Output PNG path. Default: <exp_dir>/eval_report.png")
    p.add_argument("--title", default=None,
                   help="Figure title. Default: title-cased --exp-name.")
    p.add_argument(
        "--summary",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="Compare one or more eval summary JSON files. When provided, "
        "--exp-name is not required.",
    )
    args = p.parse_args()

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit("matplotlib is required to render the eval report") from exc

    if args.summary:
        rows = []
        for item in args.summary:
            if "=" not in item:
                raise SystemExit(f"--summary must be LABEL=PATH, got: {item}")
            label, raw_path = item.split("=", 1)
            row = json.loads(Path(raw_path).read_text())
            row["label"] = label
            row.setdefault("step", label)
            rows.append(row)
    else:
        if not args.exp_name:
            raise SystemExit("--exp-name is required unless --summary is provided")
        exp_dir = Path(args.checkpoint_base_dir) / args.config_name / args.exp_name
        summary_path = exp_dir / "eval_summary.jsonl"
        if not summary_path.exists():
            raise SystemExit(f"no eval_summary.jsonl at {summary_path} yet")
        rows = [json.loads(line) for line in summary_path.read_text().splitlines() if line.strip()]
    rows = [r for r in rows if r.get("n_episodes", 0) > 0]
    if not rows:
        raise SystemExit("eval_summary.jsonl has no completed checkpoints yet (all n_episodes=0)")
    if not args.summary:
        rows.sort(key=lambda r: r["step"])

    steps = [str(r.get("label", r["step"])) for r in rows]
    success_pct = [100 * r["success_rate"] for r in rows]
    success_err = [100 * r["success_std"] for r in rows]
    fast_pct = [100 * r["fast_rate"] for r in rows]
    n = [r["n_episodes"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    fig.patch.set_facecolor("#f4f1ea")
    for ax, values, err, title, color in (
        (ax1, success_pct, success_err, "Success Rate", "#f0c75e"),
        (ax2, fast_pct, None, "FAST Rate (success < 30s)", "#8fae6b"),
    ):
        ax.set_facecolor("#f4f1ea")
        bars = ax.bar(steps, values, yerr=err, capsize=6, color=color,
                       edgecolor="black", linewidth=1.2, error_kw=dict(elinewidth=1.5))
        ax.set_title(title.upper(), fontsize=11, fontweight="bold")
        ax.set_xlabel("checkpoint step")
        ax.set_ylabel("%")
        ax.set_ylim(0, 100)
        ax.grid(axis="y", color="white", linewidth=1.2, zorder=0)
        ax.set_axisbelow(True)
        for bar, ni in zip(bars, n):
            ax.text(bar.get_x() + bar.get_width() / 2, 2, f"n={ni}",
                    ha="center", va="bottom", fontsize=8, color="#555")

    title = args.title if args.title else (
        "Eval Comparison" if args.summary else args.exp_name.replace("_", " ").title()
    )
    fig.suptitle(title, fontsize=12, fontweight="bold")
    fig.tight_layout()

    out = Path(args.out) if args.out else (
        Path("eval_report.png") if args.summary else exp_dir / "eval_report.png"
    )
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
