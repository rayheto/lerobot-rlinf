"""Read lerobot-train stdout from stdin, extract metric tokens of the form
`<key>:<value>` (loss, lr, grdn, updt_s, dataload_s, etc.), and write each
numeric one to a tensorboard SummaryWriter keyed by `step`.

Usage (piped):
    lerobot-train ... 2>&1 | tee run.log | python _sft_tb_tail.py --logdir outputs/<run>/tb

Stops when stdin closes (i.e. when lerobot-train exits).
"""
import argparse
import re
import sys

from torch.utils.tensorboard import SummaryWriter

_BIG = {"K": 1e3, "M": 1e6, "B": 1e9}
_TOKEN_RE = re.compile(r"([A-Za-z_][\w.]*):([\-+]?\d+(?:\.\d+)?(?:e[\-+]?\d+)?[KMB]?)")


def _to_num(s):
    if s[-1] in _BIG:
        return float(s[:-1]) * _BIG[s[-1]]
    return float(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", required=True)
    args = ap.parse_args()

    w = SummaryWriter(log_dir=args.logdir)
    print(f"[tb-tail] writing to {args.logdir}", file=sys.stderr, flush=True)

    last_step = -1
    for raw in sys.stdin:
        sys.stdout.write(raw)  # pass through so user still sees lerobot stdout
        sys.stdout.flush()
        if "step:" not in raw:
            continue
        toks = dict(_TOKEN_RE.findall(raw))
        if "step" not in toks:
            continue
        try:
            step = int(_to_num(toks["step"]))
        except ValueError:
            continue
        if step == last_step:
            continue
        last_step = step
        for k, v in toks.items():
            if k in {"step"}:
                continue
            try:
                w.add_scalar(k, _to_num(v), step)
            except ValueError:
                pass
        w.flush()

    w.close()


if __name__ == "__main__":
    main()
