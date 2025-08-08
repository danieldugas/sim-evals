"""
Example script for running 10 rollouts of a DROID policy on the example environment.

Usage:

First, make sure you download the simulation assets and unpack them into the root directory of this package.

Then, in a separate terminal, launch the policy server on localhost:8000 
-- make sure to set XLA_PYTHON_CLIENT_MEM_FRACTION to avoid JAX hogging all the GPU memory.

For example, to launch a pi0-FAST-DROID policy (with joint position control), 
run the command below in a separate terminal from the openpi "karl/droid_policies" branch:

XLA_PYTHON_CLIENT_MEM_FRACTION=0.5 uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_fast_droid_jointpos --policy.dir=s3://openpi-assets-simeval/pi0_fast_droid_jointpos

Finally, run the evaluation script:

python run_eval.py --episodes 10 --headless
"""

import tyro
import argparse
import gymnasium as gym
import torch
import cv2
import mediapy
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlShutdown


from src.inference.droid_jointpos import Client as DroidJointPosClient


def main(
        episodes = 10,
        headless: bool = True,
        scene: int = 1,
        ):
    # Initialize NVML
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)

    # Get memory info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # Print memory in GB
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Total: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Free: {mem_info.free / (1024 ** 3):.2f} GB")
    
    # launch omniverse app with arguments (inside function to prevent overriding tyro)
    from isaaclab.app import AppLauncher
    parser = argparse.ArgumentParser(description="Tutorial on creating an empty stage.")
    AppLauncher.add_app_launcher_args(parser)
    args_cli, _ = parser.parse_known_args()
    args_cli.enable_cameras = True
    args_cli.headless = headless
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # All IsaacLab dependent modules should be imported after the app is launched
    import src.environments # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    # Get memory info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # Print memory in GB
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Total: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Free: {mem_info.free / (1024 ** 3):.2f} GB")

    print("Creating Env")
    # Initialize the env
    env_cfg = parse_env_cfg(
        "DROID",
        device=args_cli.device,
        num_envs=1,
        use_fabric=True,
    )
    instruction = None
    match scene:
        case 1:
            instruction = "put the cube in the bowl"
        case 2:
            instruction = "put the can in the mug"
        case 3:
            instruction = "put banana in the bin"
        case _:
            raise ValueError(f"Scene {scene} not supported")
        
    env_cfg.set_scene(scene)
    env = gym.make("DROID", cfg=env_cfg)

    obs, _ = env.reset()
    obs, _ = env.reset() # need second render cycle to get correctly loaded materials
    # client = DroidJointPosClient()


    video_dir = Path("runs") / datetime.now().strftime("%Y-%m-%d") / datetime.now().strftime("%H-%M-%S")
    video_dir.mkdir(parents=True, exist_ok=True)
    video = []
    ep = 0
    max_steps = env.env.max_episode_length
    with torch.no_grad():
        for ep in range(episodes):
            for _ in tqdm(range(max_steps), desc=f"Episode {ep+1}/{episodes}"):
                # ret = client.infer(obs, "put the marker in the mug")
                external_image = obs["policy"]["external_cam"][0].clone().detach().cpu().numpy() # (720, 1280, 3) [0, 255]
                wrist_image = obs["policy"]["wrist_cam"][0].clone().detach().cpu().numpy() # (720, 1280, 3) [0, 255]
                import numpy as np
                both = np.concatenate([external_image[::4, ::4], wrist_image[::4, ::4]], axis=1)
                ret = {"action": torch.zeros(8), "viz": both}
                if not headless:
                    cv2.imshow("Right Camera", cv2.cvtColor(ret["viz"], cv2.COLOR_RGB2BGR))
                    cv2.waitKey(1)
                video.append(ret["viz"])
                action = torch.tensor(ret["action"])[None]
                obs, _, term, trunc, _ = env.step(action)
                if term or trunc:
                    break

            # client.reset()
            mediapy.write_video(
                video_dir / f"episode_{ep}.mp4",
                video,
                fps=15,
            )
            video = []

    # Get memory info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # Print memory in GB
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Total: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Free: {mem_info.free / (1024 ** 3):.2f} GB")

    env.close()
    torch.cuda.empty_cache()

    # Get memory info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # Print memory in GB
    print("After env close")
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Total: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Free: {mem_info.free / (1024 ** 3):.2f} GB")

    simulation_app.close()

    # Get memory info
    mem_info = nvmlDeviceGetMemoryInfo(handle)
    # Print memory in GB
    print(f"GPU Memory Used: {mem_info.used / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Total: {mem_info.total / (1024 ** 3):.2f} GB")
    print(f"GPU Memory Free: {mem_info.free / (1024 ** 3):.2f} GB")

if __name__ == "__main__":
    # args = tyro.cli(main)
    # run main a second time
    main(episodes=1, scene=1)
    main(episodes=1, scene=2)
    main(episodes=1, scene=3)
