"""
Quick smoke test: verify the environment loads and runs for a few steps.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from envs.pick_and_place_env import PickAndPlaceEnv

def test_env():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    xml_path = os.path.join(project_dir, "scenes", "pick_and_place_scene.xml")
    print(f"Loading scene: {xml_path}")

    for phase in [0, 1, 2]:
        print(f"\n--- Testing Phase {phase} ---")
        env = PickAndPlaceEnv(xml_path, curriculum_phase=phase)
        obs, info = env.reset()
        print(f"  Observation shape: {obs.shape}")
        print(f"  EE pos: {obs[:3]}")
        print(f"  Obj pos: {obs[3:6]}")

        # Run a few random steps
        for step in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                print(f"  Episode ended at step {step+1}, reward={reward:.3f}")
                break

        print(f"  Phase {phase} OK")
        env.close()

    print("\n=== All tests passed ===")

if __name__ == "__main__":
    test_env()
