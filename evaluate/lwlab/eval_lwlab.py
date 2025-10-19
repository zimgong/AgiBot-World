# !/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import torch
import argparse
import tqdm
import numpy as np
import requests
import json_numpy
from typing import Any, Dict

json_numpy.patch()

from lwlab.distributed.proxy import RemoteEnv
from lwlab.utils.config_loader import config_loader

from lerobot.cameras import opencv  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.train import TrainRLServerPipelineConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import make_policy
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
)
from lerobot.teleoperators import (
    gamepad,  # noqa: F401
    so101_leader,  # noqa: F401
)

from lerobot.lwrl.sim.lwlab.env_lwlab import (
    make_lwlab_robot_env, 
    make_lwlab_processors,
    step_lwlab_env_and_process_transition,
)
from lerobot.processor.converters import create_transition
from lerobot.processor.core import TransitionKey

# Based on training code analysis:
# - If no task/instruction in dataset, default_prompt is None
# - make_conversation(None) returns "What action should the robot take to None?"
# - This matches the training behavior
INSTRUCTION = ""
CTRL_FREQS = 20


class GO1Client:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port

    def predict_action(self, payload: Dict[str, Any]) -> np.ndarray:
        response = requests.post(
            f"http://{self.host}:{self.port}/act", json=payload, headers={"Content-Type": "application/json"}
        )

        if response.status_code == 200:
            result = response.json()
            action = np.array(result)
            return action
        else:
            print(f"Request failed, status code: {response.status_code}")
            print(f"Error message: {response.text}")
            return None

def parse_arguments():
    """Parse command line arguments for evaluation"""
    parser = argparse.ArgumentParser(description="LeRobot Evaluation Script")
    
    # Eval configuration
    parser.add_argument("--n_steps", type=int, default=200,
                       help="Number of steps to evaluate")
    
    # GO1Client server configuration
    parser.add_argument("--host", type=str, default="localhost", help="GO1Client server host")
    parser.add_argument("--port", type=int, default=9000, help="GO1Client server port")
    
    return parser.parse_known_args()

def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    """Converts an image to uint8 if it is a float image.

    This is important for reducing the size of the image when sending it over the network.
    """
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img

def eval_policy(env, env_processor, action_processor, go1_client, n_steps, cfg, args):
    sum_success = 0
    sum_total = 0

    obs, info = env.reset()
    env_processor.reset()
    action_processor.reset()
    # Process initial observation
    transition = create_transition(
        observation=obs,
        reward=torch.zeros((env.num_envs,), dtype=torch.float32, device=env.device),
        done=torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
        truncated=torch.zeros((env.num_envs,), dtype=torch.bool, device=env.device),
        info=info)
    transition = env_processor(transition)
    force_reinfer = True

    for _ in tqdm.tqdm(range(n_steps)):
        # Prepare observation for GO1Client (keep input unchanged as requested)
        observation = {
            k: v for k, v in transition[TransitionKey.OBSERVATION].items() if k in cfg.policy.input_features
        }
        
        # Convert torch tensors to numpy arrays for GO1Client
        payload = {}
        for k, v in observation.items():
            if isinstance(v, torch.Tensor):
                payload[k] = v.cpu().numpy()
                # to int8 if is image
                if len(v.shape) == 4 and v.shape[1] * v.shape[2] * v.shape[3] > 128 * 128:
                    payload[k] = convert_to_uint8(v.cpu().numpy())
            else:
                payload[k] = v
        
        # Add instruction and control frequency
        # For lwlab evaluation, we need to provide these fields that are missing from the dataset
        payload["instruction"] = INSTRUCTION  # Default instruction for lwlab
        payload["ctrl_freqs"] = np.array([CTRL_FREQS])  # Use ctrl_freq from config (default 20)

        # Get action from GO1Client
        # force reinfer when done or truncated
        payload["force_reinfer"] = force_reinfer
        action = go1_client.predict_action(payload)
        force_reinfer = False
        if action is None:
            print("Failed to get action from GO1Client, using zero action")
            raise Exception("Failed to get action from GO1Client")
            # action = np.zeros((env.num_envs, 6), dtype=np.float32)  # Default 6-DOF action (matching config)
        else:
            # Convert to torch tensor and ensure correct shape
            if isinstance(action, np.ndarray):
                action = torch.from_numpy(action).to(env.device)
            if action.dim() == 1:
                action = action.unsqueeze(0)  # Add batch dimension if needed

        new_transition = step_lwlab_env_and_process_transition(
            env=env,
            transition=transition,
            action=action,
            env_processor=env_processor,
            action_processor=action_processor,
        )

        done = new_transition.get(TransitionKey.DONE, torch.tensor(False))
        truncated = new_transition.get(TransitionKey.TRUNCATED, torch.tensor(False))
        info = new_transition.get(TransitionKey.INFO, {})
        reward = new_transition.get(TransitionKey.REWARD, torch.tensor(0.0, device=env.device, dtype=torch.float32))

        if info == {}:
            print("Warning: info is empty")

        num_success = info.get('is_success', torch.zeros_like(done, device=env.device, dtype=torch.bool)).sum().item()
        num_total = torch.logical_or(done, truncated).sum().item()
        sum_success += num_success
        sum_total += num_total

        if torch.any(done) or torch.any(truncated):
            force_reinfer = True
            new_transition_with_reset = new_transition
            # re-write done and truncated
            new_transition_with_reset[TransitionKey.DONE] = torch.zeros_like(done, device=env.device, dtype=torch.bool)
            new_transition_with_reset[TransitionKey.TRUNCATED] = torch.zeros_like(truncated, device=env.device, dtype=torch.bool)
            new_transition_with_reset[TransitionKey.REWARD] = torch.zeros_like(reward, device=env.device, dtype=torch.float32)
            new_transition_with_reset[TransitionKey.INFO] = {}
            #! original code will reset processor here, but skip here
            # TODO: need to implement reset processor per env index
            # env_processor.reset()
            # action_processor.reset()
            
            # recreate real transition and overwrite next observation (pass processer)
            next_observation_raw = info['final_obs']['policy'] # replace with last obs before reset
            new_transition_raw = create_transition(
                observation=next_observation_raw, info=info,
                done=torch.zeros_like(done, device=env.device, dtype=torch.bool),
                truncated=torch.zeros_like(truncated, device=env.device, dtype=torch.bool),
                reward=torch.zeros_like(reward, device=env.device, dtype=torch.float32),
            )
            # Extract values from processed transition
            new_transition = env_processor(new_transition_raw)
            # make sure those will not be used!! (only create to use processer)
            del new_transition, new_transition_raw

            info.pop('final_obs') # remove final_obs from info to save space
            
        if torch.any(done) or torch.any(truncated):
            transition = new_transition_with_reset
        else:
            transition = new_transition

    # print success rate
    print(f"Total episodes: {sum_total}")
    print(f"Success episodes: {sum_success}")
    print(f"Success rate: {sum_success / sum_total}")

        
def main(cfg: TrainRLServerPipelineConfig, args):
    """Main function"""
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Making environments...")
    env, teleop_device = make_lwlab_robot_env(cfg=cfg.env)
    env_processor, action_processor = make_lwlab_processors(env, teleop_device, cfg.env, cfg.policy.device)

    print("Initializing GO1Client...")
    go1_client = GO1Client(args.host, args.port)

    print("Evaluating with GO1Client...")
    eval_policy(env, env_processor, action_processor, go1_client=go1_client, n_steps=args.n_steps, cfg=cfg, args=args)


if __name__ == "__main__":
    import sys
    
    eval_args, remaining = parse_arguments()
    
    # temporarily hand the remaining args to the LeRobot parser-decorated entrypoint
    saved_argv = sys.argv[:]
    sys.argv = [saved_argv[0]] + remaining
    
    @parser.wrap()
    def _entry(cfg: TrainRLServerPipelineConfig):
        cfg.validate()
        main(cfg, eval_args)
    
    _entry()
