"""
Training script for 6-DOF arm pick-and-place with SAC and curriculum learning.

Key improvements over v1:
  - Loads hyperparameters from configs/sac_config.yaml
  - EvalCallback for tracking success during training
  - Custom callback for logging reward components
  - Proper warm-starting between phases
  - Saves best model based on eval reward

Usage:
    python scripts/train.py --phase 0         # REACH
    python scripts/train.py --phase 1         # GRASP (warm-starts from phase 0)
    python scripts/train.py --phase 2         # PLACE (warm-starts from phase 1)
    python scripts/train.py --phase 0 --timesteps 2000000
"""

import argparse
import os
import sys
import yaml

import numpy as np
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    CheckpointCallback, EvalCallback, CallbackList, BaseCallback
)
from stable_baselines3.common.monitor import Monitor

from envs.pick_and_place_env import PickAndPlaceEnv


PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default timesteps per phase
DEFAULT_TIMESTEPS = {0: 2_000_000, 1: 2_000_000, 2: 3_000_000}

# Number of parallel environments — more envs = more CPU simulation throughput
# = more experience fed to GPU per second
N_ENVS = 16

# Batch size — larger = more GPU work per gradient update
# 1024 fills GPU compute much better than 256 on RTX 3060
BATCH_SIZE = 1024

# Gradient steps per environment step collected
# N_ENVS * 4 means 4 gradient updates per env step = GPU works 4x harder
GRADIENT_STEPS_MULTIPLIER = 4


class RewardInfoCallback(BaseCallback):
    """Logs reward component averages to TensorBoard every N steps."""

    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self._reward_components = {}
        self._count = 0

    def _on_step(self) -> bool:
        # Collect info from all sub-environments
        infos = self.locals.get("infos", [])
        for info in infos:
            for key in ["r_baseline", "r_shaping", "r_reach_bonus",
                        "r_grasp_bonus", "r_lift_bonus", "r_place_shaping",
                        "r_place_bonus", "dist_to_obj"]:
                if key in info:
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(info[key])

            # Track success rates
            for key in ["reach_success", "grasp_success", "place_success"]:
                if key in info:
                    if key not in self._reward_components:
                        self._reward_components[key] = []
                    self._reward_components[key].append(
                        float(info[key])
                    )

        self._count += 1
        if self._count % self.log_freq == 0 and self._reward_components:
            for key, values in self._reward_components.items():
                if values:
                    self.logger.record(
                        f"reward/{key}", np.mean(values)
                    )
            self._reward_components = {}

        return True


def make_env(xml_path: str, phase: int, seed: int = 0):
    """Create environment factory function."""
    def _init():
        env = PickAndPlaceEnv(xml_path=xml_path, curriculum_phase=phase)
        env = Monitor(env, info_keywords=(
            "reach_success", "grasp_success", "place_success",
            "dist_to_obj",
        ))
        env.reset(seed=seed)
        return env
    return _init


def load_config(config_path: str) -> dict:
    """Load training config from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(phase: int, total_timesteps: int = None, log_dir: str = "logs",
          model_dir: str = "models", config_path: str = None):
    """
    Train SAC agent for a given curriculum phase.

    Args:
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        total_timesteps: Override for training timesteps
        log_dir: Directory for logs
        model_dir: Directory for saved models
        config_path: Path to YAML config (optional)
    """
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    phase_name = phase_names.get(phase, "UNKNOWN")

    # Load config if provided
    config = {}
    if config_path and os.path.exists(config_path):
        config = load_config(config_path)
        print(f"Loaded config from {config_path}")

    # Determine timesteps
    if total_timesteps is None:
        phase_config = config.get("phases", {})
        phase_key = {0: "reach", 1: "grasp", 2: "place"}.get(phase)
        if phase_key and phase_key in phase_config:
            total_timesteps = phase_config[phase_key].get(
                "total_timesteps", DEFAULT_TIMESTEPS[phase]
            )
        else:
            total_timesteps = DEFAULT_TIMESTEPS[phase]

    print(f"\n{'='*60}")
    print(f"  Training Phase {phase}: {phase_name}")
    print(f"  Timesteps: {total_timesteps:,}")
    print(f"  Parallel envs: {N_ENVS}")
    print(f"{'='*60}\n")

    # Create directories
    xml_path = os.path.join(PROJECT_DIR, "scenes", "pick_and_place_scene.xml")
    run_log_dir = os.path.join(log_dir, f"phase_{phase}")
    run_model_dir = os.path.join(model_dir, f"phase_{phase}")
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)

    # Create parallel vectorized training environments
    env = SubprocVecEnv(
        [make_env(xml_path, phase, seed=i) for i in range(N_ENVS)]
    )
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create evaluation environment (separate, single-env)
    eval_env = SubprocVecEnv(
        [make_env(xml_path, phase, seed=1000)]
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )

    # SAC hyperparameters
    sac_hp = config.get("sac_hyperparameters", {})
    sac_kwargs = {
        "policy": sac_hp.get("policy", "MlpPolicy"),
        "learning_rate": sac_hp.get("learning_rate", 3e-4),
        "buffer_size": sac_hp.get("buffer_size", 1_000_000),
        "learning_starts": sac_hp.get("learning_starts", 10_000),
        "batch_size": sac_hp.get("batch_size", BATCH_SIZE),
        "tau": sac_hp.get("tau", 0.005),
        "gamma": sac_hp.get("gamma", 0.99),
        "train_freq": (N_ENVS, "step"),
        "gradient_steps": N_ENVS * GRADIENT_STEPS_MULTIPLIER,
        "ent_coef": "auto",
        "policy_kwargs": {
            "net_arch": sac_hp.get("net_arch", [256, 256]),
            "activation_fn": nn.ReLU,
        },
        "verbose": 1,
        "tensorboard_log": run_log_dir,
        "device": "auto",
    }

    # Check for warm-start from previous phase
    prev_model_path = None
    if phase > 0:
        prev_phase_dir = os.path.join(model_dir, f"phase_{phase - 1}")
        prev_final = os.path.join(prev_phase_dir, "final_model.zip")
        if os.path.exists(prev_final):
            prev_model_path = prev_final
            print(f"  Warm-starting from: {prev_final}")

    # Initialize model
    if prev_model_path:
        model = SAC.load(
            prev_model_path.replace(".zip", ""),
            env=env,
            device="auto",
            # Reset learning rate for new phase
            learning_rate=sac_kwargs["learning_rate"],
        )
        print("  Loaded previous phase model successfully")
    else:
        model = SAC(env=env, **sac_kwargs)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=max(total_timesteps // 20, 10_000),  # ~20 checkpoints
        save_path=run_model_dir,
        name_prefix=f"sac_phase_{phase}",
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=max(total_timesteps // 40, 5_000),  # ~40 evals
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    reward_cb = RewardInfoCallback(log_freq=1000)
    callbacks = CallbackList([checkpoint_cb, eval_cb, reward_cb])

    # Train
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=(phase == 0),
    )

    # Save final model and normalization stats
    final_path = os.path.join(run_model_dir, "final_model")
    model.save(final_path)
    env.save(os.path.join(run_model_dir, "vec_normalize.pkl"))
    print(f"\nTraining complete!")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best model:  {run_model_dir}/best_model.zip")
    print(f"  VecNormalize: {run_model_dir}/vec_normalize.pkl")
    print(f"  TensorBoard:  tensorboard --logdir {run_log_dir}")

    env.close()
    eval_env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 6-DOF arm with SAC")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="Curriculum phase: 0=REACH, 1=GRASP, 2=PLACE")
    parser.add_argument("--timesteps", type=int, default=None,
                        help="Total training timesteps (default: phase-dependent)")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Model save directory")
    parser.add_argument("--config", type=str,
                        default=os.path.join(PROJECT_DIR, "configs", "sac_config.yaml"),
                        help="Path to YAML config file")

    args = parser.parse_args()
    train(
        phase=args.phase,
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        config_path=args.config,
    )
