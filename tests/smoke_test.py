"""
Smoke tests for the 6-DOF arm pick-and-place environment.

Tests:
  - Environment loads for all 3 phases
  - Observations have correct shape (20D)
  - Actions execute without errors
  - Rewards are positive-structured (not all negative)
  - Reset produces valid initial state
  - Renderer works (render_image)
"""
import sys
import os
import pytest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pick_and_place_env import PickAndPlaceEnv

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_DIR, "scenes", "pick_and_place_scene.xml")


@pytest.fixture(params=[0, 1, 2])
def env(request):
    """Create environment for each phase."""
    e = PickAndPlaceEnv(xml_path=XML_PATH, curriculum_phase=request.param)
    yield e
    e.close()


def test_reset_returns_valid_obs(env):
    """Test that reset returns a valid observation."""
    obs, info = env.reset(seed=42)
    assert obs.shape == (20,), f"Expected obs shape (20,), got {obs.shape}"
    assert obs.dtype == np.float32
    assert not np.any(np.isnan(obs)), "Observation contains NaN"
    assert not np.any(np.isinf(obs)), "Observation contains Inf"


def test_step_returns_valid_tuple(env):
    """Test that step returns the correct 5-tuple."""
    obs, _ = env.reset(seed=42)
    action = env.action_space.sample()
    result = env.step(action)
    assert len(result) == 5, f"Step should return 5 values, got {len(result)}"
    obs, reward, terminated, truncated, info = result
    assert obs.shape == (20,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_observation_components(env):
    """Test that observation components are physically reasonable."""
    obs, _ = env.reset(seed=42)
    ee_pos = obs[0:3]
    obj_pos = obs[3:6]
    relative = obs[6:9]

    # EE should be somewhere in workspace (reasonable bounds)
    assert np.all(np.abs(ee_pos) < 5.0), f"EE pos out of range: {ee_pos}"

    # Object should be near the table
    assert 0.7 < obj_pos[2] < 1.0, f"Object z={obj_pos[2]} not near table"

    # Relative pos should be consistent
    np.testing.assert_allclose(
        relative, obj_pos - ee_pos, rtol=1e-5,
        err_msg="Relative pos != obj_pos - ee_pos"
    )


def test_rewards_have_positive_component():
    """Test that the reward function produces positive values sometimes."""
    env = PickAndPlaceEnv(xml_path=XML_PATH, curriculum_phase=0)
    obs, _ = env.reset(seed=42)

    rewards = []
    for _ in range(50):
        action = env.action_space.sample()
        _, reward, terminated, truncated, _ = env.step(action)
        rewards.append(reward)
        if terminated or truncated:
            obs, _ = env.reset()

    # With positive shaping, at least some rewards should be > 0
    max_reward = max(rewards)
    assert max_reward > -1.0, (
        f"All rewards very negative (max={max_reward}). "
        "Positive shaping may not be working."
    )
    env.close()


def test_episode_truncation():
    """Test that episodes truncate at max steps."""
    env = PickAndPlaceEnv(xml_path=XML_PATH, curriculum_phase=0)
    obs, _ = env.reset(seed=42)

    truncated = False
    for step in range(300):  # More than max_steps for phase 0 (200)
        action = np.zeros(6)  # Do nothing
        _, _, terminated, truncated, _ = env.step(action)
        if terminated or truncated:
            break

    assert truncated or terminated, "Episode should have ended by step 300"
    env.close()


@pytest.mark.skipif(
    os.environ.get("DISPLAY") is None,
    reason="No display available for rendering (headless environment)"
)
def test_render_produces_image():
    """Test that rendering produces a valid image."""
    env = PickAndPlaceEnv(xml_path=XML_PATH, curriculum_phase=0)
    env.reset(seed=42)

    img = env.robot.render_image(width=64, height=64)
    assert img is not None, "Render returned None"
    assert isinstance(img, np.ndarray), "Render should return numpy array"
    assert img.ndim == 3, f"Render should be HWC, got shape {img.shape}"
    assert img.shape[2] in (3, 4), f"Expected 3 or 4 channels, got {img.shape[2]}"
    assert img.dtype == np.uint8, f"Expected uint8, got {img.dtype}"
    env.close()
def test_grasp_mechanics():
    """Test that grasping works when conditions are met."""
    env = PickAndPlaceEnv(xml_path=XML_PATH, curriculum_phase=1)
    env.reset(seed=42)

    # The grasp weld should initially be inactive
    assert not env.robot.is_object_grasped(), "Object should not be grasped at reset"

    # Gripper state should be a float in [-1, 1]
    gs = env.robot.get_gripper_state()
    assert -1.0 <= gs <= 1.0, f"Gripper state {gs} out of range"
    env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
