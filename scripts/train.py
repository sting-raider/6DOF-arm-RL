"""
Training script for 6-DOF arm pick-and-place with SAC and curriculum learning.

Usage:
    python scripts/train.py --phase 0  # REACH
    python scripts/train.py --phase 1  # GRASP
    python scripts/train.py --phase 2  # PLACE
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

from envs.pick_and_place_env import PickAndPlaceEnv
from utils.constants import MAX_EPISODE_STEPS


def make_env(phase: int, seed: int = 0):
    """Create environment factory function."""
    xml_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                            "scenes", "pick_and_place_scene.xml")

    def _init():
        env = PickAndPlaceEnv(xml_path=xml_path, curriculum_phase=phase)
        env = Monitor(env)
        return env
    return _init


def train(phase: int, total_timesteps: int = 500_000, log_dir: str = "logs",
          model_dir: str = "models"):
    """
    Train SAC agent for a given curriculum phase.

    Args:
        phase: Curriculum phase (0=REACH, 1=GRASP, 2=PLACE)
        total_timesteps: Number of training timesteps
        log_dir: Directory for logs
        model_dir: Directory for saved models
    """
    phase_names = {0: "REACH", 1: "GRASP", 2: "PLACE"}
    phase_name = phase_names.get(phase, "UNKNOWN")

    print(f"=== Training Phase {phase}: {phase_name} ===")
    print(f"Timesteps: {total_timesteps:,}")

    # Create directories
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    run_log_dir = os.path.join(log_dir, f"phase_{phase}")
    run_model_dir = os.path.join(model_dir, f"phase_{phase}")
    os.makedirs(run_log_dir, exist_ok=True)
    os.makedirs(run_model_dir, exist_ok=True)

    # Create vectorized environment
    env = DummyVecEnv([make_env(phase, seed=i) for i in range(1)])

    # Normalize observations and rewards
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # SAC hyperparameters (tuned for continuous control)
    sac_kwargs = {
        "policy": "MlpPolicy",
        "learning_rate": 3e-4,
        "buffer_size": 100_000,
        "learning_starts": 10_000,
        "batch_size": 256,
        "tau": 0.005,
        "gamma": 0.99,
        "train_freq": 1,
        "gradient_steps": 1,
        "policy_kwargs": {
            "net_arch": [256, 256],
            "activation_fn": "relu"
        },
        "verbose": 1,
        "tensorboard_log": run_log_dir,
        "device": "auto",
    }

    # Initialize SAC model
    model = SAC(env=env, **sac_kwargs)

    # Callbacks
    checkpoint_cb = CheckpointCallback(
        save_freq=50_000,
        save_path=run_model_dir,
        name_prefix=f"sac_phase_{phase}"
    )

    # Train
    print("Starting training...")
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_cb)

    # Save final model
    final_path = os.path.join(run_model_dir, "final_model")
    model.save(final_path)
    env.save(os.path.join(run_model_dir, "vec_normalize.pkl"))
    print(f"Training complete. Model saved to {final_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 6-DOF arm with SAC")
    parser.add_argument("--phase", type=int, default=0, choices=[0, 1, 2],
                        help="Curriculum phase: 0=REACH, 1=GRASP, 2=PLACE")
    parser.add_argument("--timesteps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Log directory")
    parser.add_argument("--model-dir", type=str, default="models",
                        help="Model save directory")

    args = parser.parse_args()
    train(phase=args.phase, total_timesteps=args.timesteps,
          log_dir=args.log_dir, model_dir=args.model_dir)
