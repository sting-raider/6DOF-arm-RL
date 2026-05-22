"""
6-DOF Pick-and-Place RL — Isaac Sim 5.1.0 + SB3 SAC
RTX 6000 | 95GB VRAM | MASSIVELY PARALLEL
8 parallel envs, batch_size=4096, buffer=500K
"""
import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "stable-baselines3[extra]", "gymnasium"])

from isaacsim import SimulationApp
app = SimulationApp({"headless": True})
print("Isaac Sim 5.1.0 ready", flush=True)

import gymnasium as gym; from gymnasium import spaces
import numpy as np; import torch, os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
print(f"PyTorch {torch.__version__} | CUDA {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

import omni.usd; from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf
import omni.timeline, omni.kit.app

# Scene setup — FIXES from manual_fix.py
ctx = omni.usd.get_context()
ctx.new_stage()
stage = ctx.get_stage()
stage.SetMetadata("framesPerSecond", 60)

physics_scene = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
physics_scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0, 0, -1))
physics_scene.CreateGravityMagnitudeAttr().Set(9.81)

gnd_xf = UsdGeom.Xform.Define(stage, "/World/Ground"); gnd_xf.AddTranslateOp().Set(Gf.Vec3d(0,0,-0.5))
gnd_geom = UsdGeom.Cube.Define(stage, "/World/Ground/Geom"); gnd_geom.AddScaleOp().Set(Gf.Vec3d(5,5,1))
UsdPhysics.CollisionAPI.Apply(gnd_geom.GetPrim())

tbl_xf = UsdGeom.Xform.Define(stage, "/World/Table"); tbl_xf.AddTranslateOp().Set(Gf.Vec3d(0.5,0,0.79))
tbl_geom = UsdGeom.Cube.Define(stage, "/World/Table/Geom"); tbl_geom.AddScaleOp().Set(Gf.Vec3d(0.6,0.4,0.02))
UsdPhysics.CollisionAPI.Apply(tbl_geom.GetPrim())
tbl_rb = UsdPhysics.RigidBodyAPI.Apply(tbl_xf.GetPrim()); tbl_rb.CreateKinematicEnabledAttr().Set(True)

obj_xf = UsdGeom.Xform.Define(stage, "/World/Object"); obj_xf.AddTranslateOp().Set(Gf.Vec3d(0.35,0,0.82))
obj_geom = UsdGeom.Cube.Define(stage, "/World/Object/Geom"); obj_geom.AddScaleOp().Set(Gf.Vec3d(0.04,0.04,0.04))
obj_geom.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9,0.2,0.2)])
UsdPhysics.CollisionAPI.Apply(obj_geom.GetPrim())
UsdPhysics.RigidBodyAPI.Apply(obj_xf.GetPrim())
UsdPhysics.MassAPI.Apply(obj_xf.GetPrim()).CreateMassAttr().Set(0.1)

bsk_xf = UsdGeom.Xform.Define(stage, "/World/Basket"); bsk_xf.AddTranslateOp().Set(Gf.Vec3d(0.65,0,0.80))
bsk_geom = UsdGeom.Cube.Define(stage, "/World/Basket/Geom"); bsk_geom.AddScaleOp().Set(Gf.Vec3d(0.15,0.15,0.08))
UsdPhysics.CollisionAPI.Apply(bsk_geom.GetPrim())

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetPrim().CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float, False).Set(2000.0)

timeline = omni.timeline.get_timeline_interface(); timeline.play()
kit = omni.kit.app.get_app()
print("Scene built", flush=True)

def _usd_get_pos(path):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.XformCommonAPI(prim)
    v = xf.GetTranslateAttr().Get()
    return np.array([v[0], v[1], v[2]])

def _usd_set_pos(path, pos):
    prim = stage.GetPrimAtPath(path)
    xf = UsdGeom.XformCommonAPI(prim)
    xf.SetTranslate((float(pos[0]), float(pos[1]), float(pos[2])))

# --- Env ---
N_ENVS = 8  # Parallel environments
OBJ_PATHS = [f"/World/Object_{i}" for i in range(N_ENVS)]

# Clone objects for each parallel env
for i in range(N_ENVS):
    p = f"/World/Object_{i}"
    Sdf.CopySpec(Usd.Stage.Open(stage.GetRootLayer().identifier), Sdf.Path("/World/Object"), Sdf.Path(p))
    _usd_set_pos(p, [0.35 + i*0.01, 0.0, 0.82])

class PickAndPlaceEnv(gym.Env):
    def __init__(self, env_id=0, phase=0):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(6,), dtype=np.float32)
        self.env_id = env_id; self.phase = phase
        self._steps = 0; self._prev_dist = None; self._grasped = False
        self._ee_pos = np.array([0.15, 0.0, 1.0])
        self.obj_path = f"/World/Object_{env_id}"

    def reset(self, seed=None, **kw):
        super().reset(seed=seed)
        self._steps = 0; self._grasped = False
        x = self.np_random.uniform(0.3, 0.5); y = self.np_random.uniform(-0.12, 0.12)
        _usd_set_pos(self.obj_path, [x, y, 0.82])
        self._ee_pos = np.array([0.15, 0.0, 1.0])
        for _ in range(5): kit.update()
        obs = np.concatenate([self._ee_pos, _usd_get_pos(self.obj_path)]).astype(np.float32)
        self._prev_dist = np.linalg.norm(self._ee_pos - _usd_get_pos(self.obj_path))
        return obs, {}

    def step(self, action):
        self._steps += 1; action = np.clip(action, -1, 1)
        self._ee_pos += action * 0.03
        self._ee_pos = np.clip(self._ee_pos, [0.0, -0.3, 0.5], [0.8, 0.3, 1.5])
        for _ in range(3): kit.update()

        ee = self._ee_pos; obj = _usd_get_pos(self.obj_path); dist = float(np.linalg.norm(ee - obj))
        baseline = 1.0 - min(dist / 1.0, 1.0)
        shaping = 10.0 * (self._prev_dist - dist)
        reward = baseline + shaping - 0.001

        if self.phase >= 1 and dist < 0.05:
            self._grasped = True; reward += 5.0
            if obj[2] > 0.87: reward += 2.0
        if self.phase >= 2 and self._grasped:
            basket = _usd_get_pos("/World/Basket")
            if float(np.linalg.norm(ee - basket)) < 0.15: reward += 50.0

        self._prev_dist = dist
        terminated = (self.phase == 0 and dist < 0.03)
        truncated = self._steps >= 100
        return np.concatenate([ee, obj]).astype(np.float32), reward, terminated, truncated, {"dist": dist}

# --- Train with massive parallelization ---
PHASES = {0: ("REACH", 300000), 1: ("GRASP", 300000), 2: ("PLACE", 300000)}

for phase, (name, steps) in PHASES.items():
    print(f"\n{'='*60}\nPhase {phase}: {name} — {steps} steps × {N_ENVS} envs\n{'='*60}", flush=True)
    
    def make_env(i):
        def _init():
            return PickAndPlaceEnv(env_id=i, phase=phase)
        return _init
    
    env = DummyVecEnv([make_env(i) for i in range(N_ENVS)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    model = SAC("MlpPolicy", env,
        learning_rate=3e-4, buffer_size=500000, learning_starts=10000,
        batch_size=4096, tau=0.005, gamma=0.99,
        train_freq=(N_ENVS, "step"), gradient_steps=N_ENVS*2,
        policy_kwargs={"net_arch": [256, 256]},
        verbose=1, device="cuda")
    
    os.makedirs(f"/tmp/models/phase_{phase}", exist_ok=True)
    ckpt = CheckpointCallback(save_freq=50000, save_path=f"/tmp/models/phase_{phase}", name_prefix=f"phase{phase}")
    
    model.learn(total_timesteps=steps, callback=ckpt)
    model.save(f"/tmp/models/phase_{phase}/final")
    env.save(f"/tmp/models/phase_{phase}/vecnormalize.pkl")
    print(f"Phase {phase} ({name}) COMPLETE!", flush=True)

print("\n=== ALL TRAINING DONE ===", flush=True)
app.close()
