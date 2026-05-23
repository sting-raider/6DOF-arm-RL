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
# = more experience fed to GPU per second. 32 fills 20 cores well.
N_ENVS = 32

# Batch size — larger = more GPU work per gradient update
# 4096 + 16× gradient steps = 512 updates × 4096 samples per env step
# Fills RTX 3060 compute (was 16-24% util at 2048)
BATCH_SIZE = 4096

# Gradient steps per environment step collected
# N_ENVS * 16 = 512 gradient updates per env step = 8x more GPU work
GRADIENT_STEPS_MULTIPLIER = 16
class SaveVecNormalizeCallback(BaseCallback):
    """Saves VecNormalize statistics periodically to match CheckpointCallback."""

    def __init__(self, save_freq: int, save_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            if isinstance(self.training_env, VecNormalize):
                path = os.path.join(self.save_path, "vec_normalize.pkl")
                self.training_env.save(path)
        return True


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
          model_dir: str = "models", config_path: str = None, resume_path: str = None):
    """
    Train SAC agent for a given curriculum phase.

    Args:
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        total_timesteps: Override for training timesteps
        log_dir: Directory for logs
        model_dir: Directory for saved models
        config_path: Path to YAML config (optional)
        resume_path: Path to checkpoint to resume from (optional)
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

    # Check for warm-start or resume to locate previous stats
    prev_model_path = None
    if resume_path:
        prev_model_path = resume_path
        print(f"  Resuming from checkpoint: {resume_path}")
    elif phase > 0:
        prev_phase_dir = os.path.join(model_dir, f"phase_{phase - 1}")
        prev_final = os.path.join(prev_phase_dir, "final_model.zip")
        if os.path.exists(prev_final):
            prev_model_path = prev_final
            print(f"  Warm-starting from: {prev_final}")

    # Create parallel vectorized training environments
    env = SubprocVecEnv(
        [make_env(xml_path, phase, seed=i) for i in range(N_ENVS)]
    )
    
    # Warm-start or resume VecNormalize stats if previous stats exist
    loaded_norm = False
    if prev_model_path:
        norm_path = os.path.join(os.path.dirname(prev_model_path), "vec_normalize.pkl")
        if os.path.exists(norm_path):
            print(f"  Loading VecNormalize stats from: {norm_path}")
            env = VecNormalize.load(norm_path, env)
            env.training = True
            env.norm_reward = True
            loaded_norm = True

    if not loaded_norm:
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # Create evaluation environment (separate, single-env)
    eval_env = SubprocVecEnv(
        [make_env(xml_path, phase, seed=1000)]
    )
    eval_env = VecNormalize(
        eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0
    )
    
    # Share observation normalization running stats from training to evaluation
    eval_env.obs_rms = env.obs_rms

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

    # Callbacks — use fixed intervals independent of cumulative step counter
    # NOTE: reset_num_timesteps=True below, so save_freq is relative to THIS run
    # Stable-Baselines3 callbacks check environment calls (n_calls), which equal timesteps // N_ENVS
    save_freq_steps = max(total_timesteps // 20, 10_000)
    eval_freq_steps = max(total_timesteps // 40, 5_000)
    save_freq_calls = max(save_freq_steps // N_ENVS, 1)

    checkpoint_cb = CheckpointCallback(
        save_freq=save_freq_calls,
        save_path=run_model_dir,
        name_prefix=f"sac_phase_{phase}",
    )
    save_vec_norm_cb = SaveVecNormalizeCallback(
        save_freq=save_freq_calls,
        save_path=run_model_dir,
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=run_model_dir,
        log_path=run_log_dir,
        eval_freq=max(eval_freq_steps // N_ENVS, 1),  # ~40 evals
        n_eval_episodes=10,
        deterministic=True,
        render=False,
    )
    reward_cb = RewardInfoCallback(log_freq=1000)
    callbacks = CallbackList([checkpoint_cb, save_vec_norm_cb, eval_cb, reward_cb])

    # Train
    # IMPORTANT: always reset_num_timesteps=True so CheckpointCallback and
    # EvalCallback fire relative to THIS phase's step count (not cumulative).
    # Model weights are already warm-started via SAC.load() above — the step
    # counter reset only affects callback scheduling, not the learned policy.
    print("Starting training...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        reset_num_timesteps=True,
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
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a model checkpoint to resume training from")

    args = parser.parse_args()
    train(
        phase=args.phase,
        total_timesteps=args.timesteps,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
        config_path=args.config,
        resume_path=args.resume,
    )
