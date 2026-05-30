"""Headless smoke test: boot SimulationApp, check GPU, step the sim, close."""
from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import torch
from isaacsim.core.api import World

print(f"[hello] torch={torch.__version__} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[hello] device={torch.cuda.get_device_name(0)}")

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
world.reset()
for i in range(20):
    world.step(render=False)
print(f"[hello] stepped 20 frames, sim_time={world.current_time:.3f}s")

app.close()
print("[hello] done")
