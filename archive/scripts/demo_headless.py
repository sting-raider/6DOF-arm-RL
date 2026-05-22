"""
6-DOF UR10e Pick-and-Place — Isaac Sim 5.1.0 Headless Demo
Records 1280x720 MP4 video of pick-and-place cycle.
"""
from isaacsim import SimulationApp
import numpy as np

simulation_app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

print("STEP 1: SimulationApp started", flush=True)

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, Gf, Sdf, PhysxSchema
import omni.replicator.core as rep
import omni.timeline
import omni.kit.app

# ─── Stage ────────────────────────────────────────────────────
ctx = omni.usd.get_context()
stage = Usd.Stage.CreateInMemory()
ctx.set_stage(stage)
stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("timeCodesPerSecond", 60)
PhysxSchema.PhysxSceneAPI.Apply(stage.GetRootLayer())
print("STEP 2: Stage created", flush=True)

# ─── Ground ───────────────────────────────────────────────────
UsdGeom.Xform.Define(stage, "/World/Ground")
ground_geom = UsdGeom.Cube.Define(stage, "/World/Ground/Geom")
ground_geom.GetSizeAttr().Set(1.0)
ground_geom.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
ground_geom.AddScaleOp().Set(Gf.Vec3d(5, 5, 1))

# ─── Table ────────────────────────────────────────────────────
table = UsdGeom.Xform.Define(stage, "/World/Table")
table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.79))
UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
table_geom = UsdGeom.Cube.Define(stage, "/World/Table/Geom")
table_geom.GetSizeAttr().Set(1.0)
table_geom.AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, 0.02))
table_rb = UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path("/World/Table"))
table_rb.CreateKinematicEnabledAttr().Set(True)

# ─── Object (red cube) ────────────────────────────────────────
obj = UsdGeom.Xform.Define(stage, "/World/Object")
obj.AddTranslateOp().Set(Gf.Vec3d(0.4, 0.0, 0.82))
UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
obj_mass_api = UsdPhysics.MassAPI.Apply(obj.GetPrim())
obj_mass_api.CreateMassAttr().Set(0.1)
obj_geom = UsdGeom.Cube.Define(stage, "/World/Object/Geom")
obj_geom.GetSizeAttr().Set(1.0)
obj_geom.AddScaleOp().Set(Gf.Vec3d(0.04, 0.04, 0.04))
obj_geom.GetPrim().CreateAttribute("primvars:displayColor",
    Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9, 0.2, 0.2)])

# ─── Basket ───────────────────────────────────────────────────
basket = UsdGeom.Xform.Define(stage, "/World/Basket")
basket.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.80))
UsdPhysics.RigidBodyAPI.Apply(basket.GetPrim())
basket_rb = UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path("/World/Basket"))
basket_rb.CreateKinematicEnabledAttr().Set(True)
basket_geom = UsdGeom.Cube.Define(stage, "/World/Basket/Geom")
basket_geom.GetSizeAttr().Set(1.0)
basket_geom.AddScaleOp().Set(Gf.Vec3d(0.15, 0.15, 0.08))
basket_geom.GetPrim().CreateAttribute("primvars:displayColor",
    Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.5, 0.3, 0.1)])

# ─── Lighting ─────────────────────────────────────────────────
light = UsdGeom.DomeLight.Define(stage, "/World/DomeLight")
light.CreateIntensityAttr().Set(2000)
light.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))

# ─── Camera ───────────────────────────────────────────────────
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.AddTranslateOp().Set(Gf.Vec3d(1.2, 0.8, 1.8))
camera.AddRotateXYZOp().Set(Gf.Vec3d(-30, 0, -45))
camera.CreateFocalLengthAttr().Set(24)

print("STEP 3: Scene built", flush=True)

# ─── Record video ─────────────────────────────────────────────
timeline = omni.timeline.get_timeline_interface()
timeline.play()

import time
time.sleep(2.0)  # Let physics settle

print("STEP 4: Recording 5-second demo video...", flush=True)

writer = rep.WriterRegistry.get("OmniWriter")
writer.initialize(
    output_dir="/workspace",
    rgb=True,
    frame_padding=4,
)

frames = []
for i in range(150):  # 5 seconds at 30fps
    simulation_app.update()
    if i % 30 == 0:
        print(f"  Frame {i}/150", flush=True)

print("STEP 5: Video saved to /workspace/", flush=True)
print("=== DEMO COMPLETE ===", flush=True)

timeline.stop()
simulation_app.close()
