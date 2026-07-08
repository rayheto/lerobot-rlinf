"""OpenPI-compatible websocket server for pi05 + RLinf residual MLP eval."""
from __future__ import annotations

import argparse
import asyncio
import http
import logging
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import websockets
import websockets.frames
from websockets.server import serve
from omegaconf import OmegaConf
from openpi_client import msgpack_numpy
from openpi_client.websocket_client_policy import WebsocketClientPolicy

from src.rl.rlinf_residual.model import build_model


def _find_full_weights(ckpt_dir: Path) -> Path:
    candidates = [
        ckpt_dir / "actor" / "model_state_dict" / "full_weights.pt",
        ckpt_dir / "model_state_dict" / "full_weights.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise SystemExit(
        "RLinf residual checkpoint is missing full weights. Tried: "
        + ", ".join(str(p) for p in candidates)
    )


def _load_policy(args: argparse.Namespace):
    cfg = OmegaConf.create(
        {
            "obs_dim": args.obs_dim,
            "action_dim": args.action_dim,
            "num_action_chunks": 1,
            "add_value_head": True,
            "add_q_head": False,
            "q_head_type": "default",
            "init_log_std": -1.0,
        }
    )
    model = build_model(cfg, torch_dtype=torch.float32)
    weights_path = _find_full_weights(Path(args.rl_checkpoint_dir).resolve())
    state_dict = torch.load(weights_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        logging.warning(
            "loaded residual policy with missing=%s unexpected=%s",
            missing,
            unexpected,
        )
    model.eval()
    return model


class ResidualPolicy:
    def __init__(self, args: argparse.Namespace):
        self.base = WebsocketClientPolicy(host=args.base_host, port=args.base_port)
        self.model = _load_policy(args)
        self.device = torch.device(args.device)
        self.model.to(self.device)
        self.residual_clip = float(args.residual_clip)

    @torch.inference_mode()
    def infer(self, obs: dict) -> dict:
        base_result = self.base.infer(obs)
        base_actions = np.asarray(base_result["actions"], dtype=np.float32)
        if base_actions.ndim != 2 or base_actions.shape[-1] != 6:
            raise RuntimeError(f"unexpected base action shape: {base_actions.shape}")

        state = np.asarray(obs["state"], dtype=np.float32).reshape(1, -1)
        if state.shape[-1] != 6:
            raise RuntimeError(f"unexpected state shape: {state.shape}")

        states = np.repeat(state, base_actions.shape[0], axis=0)
        model_states = np.concatenate([states, base_actions], axis=-1)
        model_obs = {
            "states": torch.as_tensor(
                model_states, dtype=torch.float32, device=self.device
            )
        }
        residual, _ = self.model.predict_action_batch(
            model_obs,
            calculate_logprobs=False,
            calculate_values=False,
            return_obs=False,
            mode="eval",
        )
        residual_np = residual[:, 0, :].detach().cpu().numpy().astype(np.float32)
        if self.residual_clip > 0:
            residual_np = np.clip(residual_np, -self.residual_clip, self.residual_clip)

        out = dict(base_result)
        out["actions"] = (base_actions + residual_np).astype(np.float32)
        out["base_actions"] = base_actions
        out["residual_actions"] = residual_np
        return out


def _health_check(path, _request_headers):
    if path == "/healthz":
        return http.HTTPStatus.OK, [], b"OK\n"
    return None


async def _handler(websocket, policy: ResidualPolicy):
    logging.info("connection from %s opened", websocket.remote_address)
    packer = msgpack_numpy.Packer()
    await websocket.send(packer.pack({"backend": "rlinf-residual"}))
    prev_total_time = None
    while True:
        try:
            start = time.monotonic()
            obs = msgpack_numpy.unpackb(await websocket.recv())
            infer_start = time.monotonic()
            action = policy.infer(obs)
            infer_ms = (time.monotonic() - infer_start) * 1000
            action["server_timing"] = {"infer_ms": infer_ms}
            if prev_total_time is not None:
                action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
            await websocket.send(packer.pack(action))
            prev_total_time = time.monotonic() - start
        except websockets.ConnectionClosed:
            logging.info("connection from %s closed", websocket.remote_address)
            break
        except Exception:
            await websocket.send(traceback.format_exc())
            await websocket.close(
                code=websockets.frames.CloseCode.INTERNAL_ERROR,
                reason="Internal server error. Traceback included in previous frame.",
            )
            raise


async def _run(args: argparse.Namespace):
    policy = ResidualPolicy(args)
    async with serve(
        lambda websocket: _handler(websocket, policy),
        args.host,
        args.port,
        compression=None,
        max_size=None,
        process_request=_health_check,
    ) as server:
        logging.info("rlinf residual server listening on %s:%s", args.host, args.port)
        await server.serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-checkpoint-dir", required=True)
    parser.add_argument("--base-host", default="127.0.0.1")
    parser.add_argument("--base-port", type=int, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--obs-dim", type=int, default=12)
    parser.add_argument("--action-dim", type=int, default=6)
    parser.add_argument("--residual-clip", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
