"""
Isaac Sim 5.1.0 + SB3 SAC — Reach Task (200K steps on RTX 6000)
Positive reward shaping — the fix we know works.
"""
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})
print("Isaac Sim ready", flush=True)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import os

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

# ─── Build Scene ──────────────────────────────────────────────
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.timeline
import omni.kit.app

ctx = omni.usd.get_context()
stage = Usd.Stage.CreateInMemory()
ctx.set_stage(stage)
stage.SetMetadata("timeCodesPerSecond", 60)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetRootLayer())

# Ground
UsdGeom.Cube.Define(stage, "/World/Ground/Geom").AddScaleOp().Set(Gf.Vec3d(5, 5, 1))
UsdGeom.Xform.Define(stage, "/World/Ground").AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))

# Table  
table = UsdGeom.Xform.Define(stage, "/World/Table")
table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.79))
UsdGeom.Cube.Define(stage, "/World/Table/Geom").AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, 0.02))
UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path("/World/Table")).CreateKinematicEnabledAttr().Set(True)

# Object (red cube with physics)
obj = UsdGeom.Xform.Define(stage, "/World/Object")
obj.AddTranslateOp().Set(Gf.Vec3d(0.35, 0.0, 0.82))
obj_geom = UsdGeom.Cube.Define(stage, "/World/Object/Geom")
obj_geom.AddScaleOp().Set(Gf.Vec3d(0.04, 0.04, 0.04))
obj_geom.GetPrim().CreateAttribute("primvars:displayColor",
    Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9, 0.2, 0.2)])
UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
UsdPhysics.MassAPI.Apply(obj.GetPrim()).CreateMassAttr().Set(0.1)

# Basket
basket = UsdGeom.Xform.Define(stage, "/World/Basket")
basket.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.80))
UsdGeom.Cube.Define(stage, "/World/Basket/Geom").AddScaleOp().Set(Gf.Vec3d(0.15, 0.15, 0.08))

# Light
light = UsdGeom.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr().Set(2000)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
app = omni.kit.app.get_app()
print("Scene + physics running", flush=True)

# ─── RL Environment ───────────────────────────────────────────
class ReachEnv(gym.Env):
    """Simple reach task: move EE (simplified as a free body) to object."""
    
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)  # x,y,z delta
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)  # ee(3)+obj(3)
        self._step_count = 0
        self._prev_dist = None
        
        # Access prims
        self.obj_prim = stage.GetPrimAtPath("/World/Object")
        self.ee_prim = stage.GetPrimAtPath("/World/Ground")  # Start EE at ground
        
    def _get_obj_pos(self):
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(self.obj_prim)
        return np.array(xform.GetLocalTransformation().ExtractTranslation())
    
    def _set_obj_pos(self, pos):
        from pxr import UsdGeom
        xform = UsdGeom.Xformable(self.obj_prim)
        xform.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
    
    def reset(self, seed=None, **kwargs):
        super().reset(seed=seed)
        self._step_count = 0
        
        # Randomize object on table
        x = self.np_random.uniform(0.3, 0.5)
        y = self.np_random.uniform(-0.12, 0.12)
        self._set_obj_pos([x, y, 0.82])
        self._ee_target = np.array([0.4, 0.0, 0.95])
        self._ee_pos = np.array([0.2, 0.0, 1.0])
        
        for _ in range(5):
            app.update()
        
        obs = np.concatenate([self._ee_pos, self._get_obj_pos()]).astype(np.float32)
        self._prev_dist = np.linalg.norm(self._ee_pos - self._get_obj_pos())
        return obs, {}
    
    def step(self, action):
        self._step_count += 1
        action = np.clip(action, -1, 1)
        
        # Move EE toward action
        self._ee_pos = self._ee_pos + action * 0.03
        self._ee_pos = np.clip(self._ee_pos, [0, -0.3, 0.6], [0.8, 0.3, 1.5])
        
        for _ in range(3):
            app.update()
        
        obj_pos = self._get_obj_pos()
        dist = float(np.linalg.norm(self._ee_pos - obj_pos))
        
        # POSITIVE reward shaping
        baseline = 1.0 - min(dist / 1.0, 1.0)
        shaping = 10.0 * (self._prev_dist - dist)
        reward = baseline + shaping - 0.001
        
        self._prev_dist = dist
        terminated = dist < 0.05
        truncated = self._step_count >= 100
        
        info = {"dist": dist, "is_success": bool(terminated)}
        return np.concatenate([self._ee_pos, obj_pos]).astype(np.float32), reward, terminated, truncated, info
    
    def close(self):
        pass

# ─── Train ────────────────────────────────────────────────────
print("Creating env...", flush=True)
env = ReachEnv()
env = Monitor(env)

model = SAC(
    "MlpPolicy", env,
    learning_rate=3e-4,
    buffer_size=50000,
    learning_starts=5000,
    batch_size=256,
    tau=0.005,
    gamma=0.99,
    policy_kwargs={"net_arch": [256, 256]},
    verbose=1,
    device="cuda",
    tensorboard_log="/workspace/logs",
)

os.makedirs("/workspace/models", exist_ok=True)
ckpt = CheckpointCallback(save_freq=50000, save_path="/workspace/models", name_prefix="reach")

print(f"Training 200K steps on {torch.cuda.get_device_name(0)}...", flush=True)
model.learn(total_timesteps=200000, callback=ckpt)
model.save("/workspace/models/reach_final")
print("DONE! Model saved.", flush=True)

env.close()
simulation_app.close()
