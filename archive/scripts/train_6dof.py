"""
6-DOF UR10e Pick-and-Place RL — Isaac Sim 5.1.0 + SB3 SAC
RTX 6000 Blackwell | 95GB VRAM | Positive Reward Shaping
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3[extra]", "gymnasium"])

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("Isaac Sim 5.1.0 ready", flush=True)

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch, os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

# --- Build Scene in Isaac Sim ---
import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.timeline, omni.kit.app

ctx = omni.usd.get_context()
stage = Usd.Stage.CreateInMemory()
PhysxSchema.PhysxSceneAPI.Apply(stage.GetRootLayer())
stage.SetMetadata("timeCodesPerSecond", 60)

# Ground
UsdGeom.Xform.Define(stage, "/World/Ground").AddTranslateOp().Set(Gf.Vec3d(0,0,-0.5))
UsdGeom.Cube.Define(stage, "/World/Ground/Geom").AddScaleOp().Set(Gf.Vec3d(5,5,1))

# Table (0.79m height, kinematic)
t = UsdGeom.Xform.Define(stage, "/World/Table"); t.AddTranslateOp().Set(Gf.Vec3d(0.5,0,0.79))
UsdGeom.Cube.Define(stage, "/World/Table/Geom").AddScaleOp().Set(Gf.Vec3d(0.6,0.4,0.02))
UsdPhysics.RigidBodyAPI.Apply(t.GetPrim())
UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path("/World/Table")).CreateKinematicEnabledAttr().Set(True)

# Object (red cube, dynamic)
o = UsdGeom.Xform.Define(stage, "/World/Object"); o.AddTranslateOp().Set(Gf.Vec3d(0.35,0,0.82))
og = UsdGeom.Cube.Define(stage, "/World/Object/Geom"); og.AddScaleOp().Set(Gf.Vec3d(0.04,0.04,0.04))
og.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9,0.2,0.2)])
UsdPhysics.RigidBodyAPI.Apply(o.GetPrim())
UsdPhysics.MassAPI.Apply(o.GetPrim()).CreateMassAttr().Set(0.1)

# Basket (target for place task)
b = UsdGeom.Xform.Define(stage, "/World/Basket"); b.AddTranslateOp().Set(Gf.Vec3d(0.65,0,0.80))
UsdGeom.Cube.Define(stage, "/World/Basket/Geom").AddScaleOp().Set(Gf.Vec3d(0.15,0.15,0.08))

UsdGeom.DomeLight.Define(stage, "/World/DomeLight").CreateIntensityAttr().Set(2000)

timeline = omni.timeline.get_timeline_interface(); timeline.play()
kit = omni.kit.app.get_app()
print("Scene built", flush=True)

# --- 6-DOF Pick-and-Place RL Environment ---
class PickAndPlaceEnv(gym.Env):
    """6-DOF arm reaching a target object on a table. EE is a free-floating proxy."""
    
    def __init__(self, phase=0):
        super().__init__()
        # Action: [dx, dy, dz] for end-effector movement
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        # Obs: [ee_x, ee_y, ee_z, obj_x, obj_y, obj_z]
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.phase = phase  # 0=REACH, 1=GRASP+LIFT, 2=PLACE
        self._steps = 0
        self._prev_dist = None
        self._grasped = False
        self._ee_pos = np.array([0.15, 0.0, 1.0])  # Start position

    def _get_obj_pos(self):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object"))
        p = xf.GetLocalTransformation().ExtractTranslation()
        return np.array([p[0], p[1], p[2]])

    def _set_obj_pos(self, pos):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Object"))
        xf.AddTranslateOp().Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

    def _get_basket_pos(self):
        xf = UsdGeom.Xformable(stage.GetPrimAtPath("/World/Basket"))
        p = xf.GetLocalTransformation().ExtractTranslation()
        return np.array([p[0], p[1], p[2]])

    def reset(self, seed=None, **kw):
        super().reset(seed=seed)
        self._steps = 0
        self._grasped = False
        # Randomize object on table
        x = self.np_random.uniform(0.3, 0.5)
        y = self.np_random.uniform(-0.12, 0.12)
        self._set_obj_pos([x, y, 0.82])
        self._ee_pos = np.array([0.15, 0.0, 1.0])
        for _ in range(5): kit.update()
        obs = self._get_obs()
        self._prev_dist = np.linalg.norm(self._ee_pos - self._get_obj_pos())
        return obs, {}

    def _get_obs(self):
        return np.concatenate([self._ee_pos, self._get_obj_pos()]).astype(np.float32)

    def step(self, action):
        self._steps += 1
        action = np.clip(action, -1, 1)
        # Move EE by action
        self._ee_pos += action * 0.03
        self._ee_pos = np.clip(self._ee_pos, [0.0, -0.3, 0.5], [0.8, 0.3, 1.5])
        for _ in range(3): kit.update()

        ee = self._ee_pos
        obj = self._get_obj_pos()
        dist = float(np.linalg.norm(ee - obj))
        basket = self._get_basket_pos()

        # --- CURRICULUM REWARDS ---
        # Phase 0: REACH — get EE close to object
        baseline = 1.0 - min(dist / 1.0, 1.0)  # 1.0 at dist=0
        shaping = 10.0 * (self._prev_dist - dist)
        reward = baseline + shaping - 0.001

        # Phase 1: REACH + GRASP (simulate grasp when very close)
        if self.phase >= 1 and dist < 0.05:
            self._grasped = True
            reward += 5.0  # Grasp bonus
            if obj[2] > 0.87:  # Lifted above table
                reward += 2.0

        # Phase 2: REACH + GRASP + PLACE
        if self.phase >= 2 and self._grasped:
            dist_to_basket = float(np.linalg.norm(ee - basket))
            if dist_to_basket < 0.15:
                reward += 50.0  # Place bonus!

        self._prev_dist = dist
        terminated = False
        if self.phase == 0:
            terminated = dist < 0.03
        elif self.phase == 2 and self._grasped:
            terminated = float(np.linalg.norm(ee - basket)) < 0.10

        truncated = self._steps >= 100
        info = {"dist": dist, "grasped": self._grasped}
        return self._get_obs(), reward, terminated, truncated, info

# --- Train All Phases ---
PHASES = {0: ("REACH", 200000), 1: ("GRASP", 200000), 2: ("PLACE", 200000)}

for phase, (name, steps) in PHASES.items():
    print(f"\n{'='*60}\nPhase {phase}: {name} — {steps} steps\n{'='*60}", flush=True)
    
    env = PickAndPlaceEnv(phase=phase)
    env = Monitor(env)
    
    model = SAC("MlpPolicy", env,
        learning_rate=3e-4, buffer_size=100000, learning_starts=10000,
        batch_size=256, tau=0.005, gamma=0.99,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1, device="cuda", tensorboard_log="/tmp/logs")
    
    os.makedirs(f"/tmp/models/phase_{phase}", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=50000, save_path=f"/tmp/models/phase_{phase}", name_prefix=f"phase{phase}")
    
    model.learn(total_timesteps=steps, callback=ckpt)
    model.save(f"/tmp/models/phase_{phase}/final")
    print(f"Phase {phase} ({name}) COMPLETE!", flush=True)

print("\n=== ALL TRAINING DONE ===", flush=True)
app.close()
