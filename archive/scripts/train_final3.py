"""
6-DOF Pick-and-Place RL — Isaac Sim 5.1.0 + SB3 SAC | RTX 6000
ALL FIXES APPLIED + MASSIVE PARALLELIZATION
16 envs, batch_size=8192, buffer=1M, net=[512,512], warm-start
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
print(f"PyTorch {torch.__version__} | CUDA: {torch.cuda.is_available()} | {torch.cuda.get_device_name(0)}", flush=True)

import omni.usd; import omni.timeline; import omni.kit.app
from pxr import Usd, UsdGeom, UsdPhysics, UsdLux, Gf, Sdf

# --- FIXES 1-6: Proper scene setup ---
ctx = omni.usd.get_context(); ctx.new_stage(); stage = ctx.get_stage()
stage.SetMetadata("framesPerSecond", 60)

ps = UsdPhysics.Scene.Define(stage, "/World/PhysicsScene")
ps.CreateGravityDirectionAttr().Set(Gf.Vec3f(0,0,-1)); ps.CreateGravityMagnitudeAttr().Set(9.81)

gx = UsdGeom.Xform.Define(stage, "/World/Ground"); gx.AddTranslateOp().Set(Gf.Vec3d(0,0,-0.5))
gg = UsdGeom.Cube.Define(stage, "/World/Ground/Geom"); gg.AddScaleOp().Set(Gf.Vec3d(5,5,1))
UsdPhysics.CollisionAPI.Apply(gg.GetPrim())

tx = UsdGeom.Xform.Define(stage, "/World/Table"); tx.AddTranslateOp().Set(Gf.Vec3d(0.5,0,0.79))
tg = UsdGeom.Cube.Define(stage, "/World/Table/Geom"); tg.AddScaleOp().Set(Gf.Vec3d(0.6,0.4,0.02))
UsdPhysics.CollisionAPI.Apply(tg.GetPrim())
tr = UsdPhysics.RigidBodyAPI.Apply(tx.GetPrim()); tr.CreateKinematicEnabledAttr().Set(True)

ox = UsdGeom.Xform.Define(stage, "/World/Object"); ox.AddTranslateOp().Set(Gf.Vec3d(0.35,0,0.82))
og = UsdGeom.Cube.Define(stage, "/World/Object/Geom"); og.AddScaleOp().Set(Gf.Vec3d(0.04,0.04,0.04))
og.GetPrim().CreateAttribute("primvars:displayColor", Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9,0.2,0.2)])
UsdPhysics.CollisionAPI.Apply(og.GetPrim()); UsdPhysics.RigidBodyAPI.Apply(ox.GetPrim())
UsdPhysics.MassAPI.Apply(ox.GetPrim()).CreateMassAttr().Set(0.1)

bx = UsdGeom.Xform.Define(stage, "/World/Basket"); bx.AddTranslateOp().Set(Gf.Vec3d(0.65,0,0.80))
bg = UsdGeom.Cube.Define(stage, "/World/Basket/Geom"); bg.AddScaleOp().Set(Gf.Vec3d(0.15,0.15,0.08))
UsdPhysics.CollisionAPI.Apply(bg.GetPrim())

dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome.GetPrim().CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float, False).Set(2000.0)

timeline = omni.timeline.get_timeline_interface(); timeline.play()
kit = omni.kit.app.get_app()
print("Scene built", flush=True)

# --- FIX 7: Safe USD helpers using XformCommonAPI ---
def _usd_get_pos(path):
    prim = stage.GetPrimAtPath(path); api = UsdGeom.XformCommonAPI(prim)
    t, *_ = api.GetXformVectors(Usd.TimeCode.Default())
    return np.array([t[0], t[1], t[2]], dtype=np.float64)

def _usd_set_pos(path, pos):
    prim = stage.GetPrimAtPath(path); api = UsdGeom.XformCommonAPI(prim)
    api.SetTranslate(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))

# --- Env with all fixes ---
class PickAndPlaceEnv(gym.Env):
    def __init__(self, phase=0):
        super().__init__()
        self.action_space = spaces.Box(-1,1,shape=(3,),dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf,np.inf,shape=(6,),dtype=np.float32)
        self.phase = phase; self._steps = 0; self._prev_dist = None
        self._grasped = False; self._ee_pos = np.array([0.15,0.0,1.0], dtype=np.float64)

    def reset(self, seed=None, **kw):
        super().reset(seed=seed); self._steps = 0; self._grasped = False
        x = self.np_random.uniform(0.30,0.50); y = self.np_random.uniform(-0.12,0.12)
        _usd_set_pos("/World/Object", [x,y,0.82])
        self._ee_pos = np.array([0.15,0.0,1.0], dtype=np.float64)
        for _ in range(5): kit.update()
        obs = np.concatenate([self._ee_pos, _usd_get_pos("/World/Object")]).astype(np.float32)
        self._prev_dist = float(np.linalg.norm(self._ee_pos - _usd_get_pos("/World/Object")))
        return obs, {}

    def step(self, action):
        self._steps += 1; action = np.clip(action,-1,1)
        self._ee_pos += action * 0.03
        self._ee_pos = np.clip(self._ee_pos, [0.0,-0.30,0.50], [0.80,0.30,1.50])
        # FIX 8: When grasped, keep object at EE
        if self._grasped: _usd_set_pos("/World/Object", self._ee_pos)
        for _ in range(3): kit.update()

        ee = self._ee_pos; obj = _usd_get_pos("/World/Object")
        basket = _usd_get_pos("/World/Basket"); dist = float(np.linalg.norm(ee - obj))

        # Reward (positive shaping)
        baseline = 1.0 - min(dist/1.0, 1.0); shaping = 10.0*(self._prev_dist - dist)
        reward = baseline + shaping - 0.001

        if self.phase >= 1 and dist < 0.05:
            if not self._grasped: self._grasped = True; reward += 5.0
            if obj[2] > 0.87: reward += 2.0
        # FIX 9: Place reward uses object→basket distance
        if self.phase >= 2 and self._grasped:
            dob = float(np.linalg.norm(obj - basket))
            reward += 5.0*(1.0 - min(dob/1.0, 1.0))
            if dob < 0.15: reward += 50.0

        self._prev_dist = dist
        terminated = False
        if self.phase == 0: terminated = dist < 0.03
        elif self.phase == 2 and self._grasped:
            terminated = float(np.linalg.norm(obj - basket)) < 0.10
        truncated = self._steps >= 100
        return np.concatenate([ee,obj]).astype(np.float32), float(reward), terminated, truncated, {"dist":dist}

# --- MASSIVELY PARALLEL TRAINING ---
N_ENVS = 16
PHASES = {0:("REACH",300000), 1:("GRASP",300000), 2:("PLACE",300000)}

def _make_env(p): return Monitor(PickAndPlaceEnv(phase=p))
prev = None
for phase, (name, steps) in PHASES.items():
    print(f"\n{'='*60}\nPhase {phase}: {name} — {steps}×{N_ENVS} envs, batch=8192\n{'='*60}", flush=True)
    env = DummyVecEnv([lambda p=phase: _make_env(p)] * N_ENVS)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    os.makedirs(f"/tmp/models/phase_{phase}", exist_ok=True)

    # FIX 11: Warm-start
    if prev:
        print(f"  Warm-start from phase {phase-1}", flush=True)
        model = SAC.load(prev, env=env, device="cuda")
    else:
        model = SAC("MlpPolicy", env, learning_rate=3e-4, buffer_size=1000000,
            learning_starts=10000, batch_size=8192, tau=0.005, gamma=0.99,
            train_freq=(N_ENVS,"step"), gradient_steps=N_ENVS,
            policy_kwargs={"net_arch":[512,512]}, verbose=1, device="cuda")

    ckpt = CheckpointCallback(save_freq=100000, save_path=f"/tmp/models/phase_{phase}", name_prefix=f"phase{phase}")
    model.learn(total_timesteps=steps, callback=ckpt, reset_num_timesteps=(phase==0))
    prev = f"/tmp/models/phase_{phase}/final"
    model.save(prev); env.save(f"/tmp/models/phase_{phase}/vecnorm.pkl")
    print(f"Phase {phase} ({name}) DONE!", flush=True)

print("\n=== ALL TRAINING DONE ===", flush=True)
app.close()
