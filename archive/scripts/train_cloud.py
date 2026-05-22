"""
6-DOF UR10e Pick-and-Place RL Training — Isaac Sim 5.1.0 + SB3 SAC
Phase 0: REACH — 200K steps (fast test on RTX 6000)
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True, "width": 640, "height": 480})
print("Isaac Sim ready", flush=True)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor

print(f"PyTorch: {torch.__version__} | CUDA: {torch.cuda.is_available()} | GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ─── Build Scene in Isaac Sim ─────────────────────────────────
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema

ctx = omni.usd.get_context()
stage = Usd.Stage.CreateInMemory()
ctx.set_stage(stage)
stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("timeCodesPerSecond", 60)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetRootLayer())

# Ground
UsdGeom.Cube.Define(stage, "/World/Ground/Geom").AddScaleOp().Set(Gf.Vec3d(5, 5, 1))
ground = UsdGeom.Xform.Define(stage, "/World/Ground")
ground.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))

# Table
table = UsdGeom.Xform.Define(stage, "/World/Table")
table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.79))
table_geom = UsdGeom.Cube.Define(stage, "/World/Table/Geom")
table_geom.AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, 0.02))
UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path("/World/Table")).CreateKinematicEnabledAttr().Set(True)

# Object
obj = UsdGeom.Xform.Define(stage, "/World/Object")
obj.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.82))
UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr().Set(0.1)
obj_geom = UsdGeom.Cube.Define(stage, "/World/Object/Geom")
obj_geom.AddScaleOp().Set(Gf.Vec3d(0.04, 0.04, 0.04))

# Basket
basket = UsdGeom.Xform.Define(stage, "/World/Basket")
basket.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.80))
basket_geom = UsdGeom.Cube.Define(stage, "/World/Basket/Geom")
basket_geom.AddScaleOp().Set(Gf.Vec3d(0.15, 0.15, 0.08))

# Light
light = UsdGeom.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr().Set(2000)

print("Scene built", flush=True)

# ─── Simple RL Environment (no Isaac Lab) ─────────────────────
class UR10eReachEnv(gym.Env):
    """Simplified UR10e reach environment using direct joint control."""
    
    def __init__(self, headless=True):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(6,), dtype=np.float32)
        self.obs_dim = 8  # EE pos(3) + obj pos(3) + gripper(1) + grasping(1)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self._step_count = 0
        self._prev_dist = None
        
        # Start physics
        import omni.timeline
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        self._app = omni.kit.app.get_app()
        
        # Get prims
        self.obj_prim = stage.GetPrimAtPath("/World/Object")
        self.basket_prim = stage.GetPrimAtPath("/World/Basket")
        
    def _get_obj_pos(self):
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(self.obj_prim)
        return np.array(xform.GetLocalTransformation().ExtractTranslation())
    
    def _get_basket_pos(self):
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(self.basket_prim)
        return np.array(xform.GetLocalTransformation().ExtractTranslation())
    
    def _get_ee_pos(self):
        # Simplified: use object pos as proxy for EE (no robot loaded yet)
        # In full version, read from UR10e articulation
        return np.array([0.35, 0.0, 1.0])  # Default EE position
    
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._step_count = 0
        
        # Randomize object position
        x = np.random.uniform(0.25, 0.45)
        y = np.random.uniform(-0.15, 0.15)
        z = 0.82
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(self.obj_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(x, y, z))
        
        # Step physics a few times
        for _ in range(10):
            self._app.update()
        
        obs = self._get_obs()
        self._prev_dist = np.linalg.norm(self._get_ee_pos()[:2] - np.array([x, y]))
        return obs, {}
    
    def _get_obs(self):
        ee = self._get_ee_pos()
        obj = self._get_obj_pos()
        return np.concatenate([ee, obj, [0.0], [0.0]]).astype(np.float32)
    
    def step(self, action):
        self._step_count += 1
        
        # Move EE toward action direction (simplified)
        action = np.clip(action, -1, 1)
        movement = action[:3] * 0.02  # 2cm per step
        current_ee = self._get_ee_pos()
        new_ee = current_ee + np.array([movement[0], movement[1], movement[2]])
        
        # Step physics
        for _ in range(5):
            self._app.update()
        
        obs = self._get_obs()
        ee = self._get_ee_pos()
        obj = self._get_obj_pos()
        dist = float(np.linalg.norm(ee[:2] - obj[:2]))
        
        # POSITIVE reward shaping (the fix!)
        baseline = 1.0 - min(dist / 1.0, 1.0)  # 1.0 at dist=0, 0.0 at dist>=1
        shaping = 20.0 * (self._prev_dist - dist)  # Improvement reward
        reward = baseline + shaping - 0.001  # Small step penalty
        self._prev_dist = dist
        
        terminated = dist < 0.05  # Reached!
        truncated = self._step_count >= 100
        
        info = {"dist": dist, "is_success": terminated}
        return obs, reward, terminated, truncated, info
    
    def close(self):
        self.timeline.stop()

# ─── Train ────────────────────────────────────────────────────
print("Creating environment...", flush=True)
env = UR10eReachEnv()
env = Monitor(env)

print("Setting up SAC...", flush=True)
model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=50000,
    learning_starts=5000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    train_freq=1,
    gradient_steps=1,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    device="cuda",
    tensorboard_log="/workspace/logs",
)

os.makedirs("/workspace/models", exist_ok=True)
checkpoint = CheckpointCallback(
    save_freq=25000,
    save_path="/workspace/models",
    name_prefix="ur10e_reach"
)

print("Training 200K steps on RTX 6000...", flush=True)
model.learn(total_timesteps=200000, callback=checkpoint)
model.save("/workspace/models/ur10e_reach_final")
print("Training complete! Model saved.", flush=True)

env.close()
simulation_app.close()
