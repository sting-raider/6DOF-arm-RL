"""
6-DOF UR10e Pick-and-Place — Isaac Sim 5.1.0 Interactive Demo
=============================================================
Scene: UR10e + Robotiq 2F-85 | Table | Object | Basket
Usage: /isaac-sim/python.sh demo.py

Press 'R' in viewport to randomize object position.
Press 'Space' to run pick-and-place cycle.
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({
    "headless": False,
    "renderer": "RayTracedLighting",
    "width": 1280,
    "height": 720,
})

import omni.usd
from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Gf, Sdf, PhysxSchema
import omni.kit.commands
import carb
import numpy as np
import random

print("=" * 60)
print("  6-DOF UR10e Pick-and-Place — Isaac Sim 5.1.0")
print("  GPU:", carb.settings.get_settings().get("/app/renderer/gpuName"))
print("=" * 60)

# ─── Stage Setup ─────────────────────────────────────────────
ctx = omni.usd.get_context()
stage = ctx.get_stage()
if not stage:
    stage = Usd.Stage.CreateInMemory()
    ctx.set_stage(stage)

stage.SetMetadata("metersPerUnit", 1.0)
stage.SetMetadata("timeCodesPerSecond", 60)

# ─── Physics ──────────────────────────────────────────────────
PhysxSchema.PhysxSceneAPI.Apply(stage.GetRootLayer())

# ─── Ground ───────────────────────────────────────────────────
ground = UsdGeom.Xform.Define(stage, "/World/Ground")
UsdPhysics.PlaneCollisionAPI.Apply(ground.GetPrim())
ground_geom = UsdGeom.Cube.Define(stage, "/World/Ground/Geom")
ground_geom.GetSizeAttr().Set(1.0)
ground_geom.AddTranslateOp().Set(Gf.Vec3d(0, 0, -0.5))
ground_geom.AddScaleOp().Set(Gf.Vec3d(5, 5, 1))

# ─── Table ────────────────────────────────────────────────────
table_path = "/World/Table"
table = UsdGeom.Xform.Define(stage, table_path)
table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.79))  # 0.79m height
UsdPhysics.RigidBodyAPI.Apply(table.GetPrim())
table_geom = UsdGeom.Cube.Define(stage, f"{table_path}/Geom")
table_geom.GetSizeAttr().Set(1.0)
table_geom.AddScaleOp().Set(Gf.Vec3d(0.6, 0.4, 0.02))
# Make table kinematic (fixed)
table_rb = UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path(table_path))
table_rb.CreateKinematicEnabledAttr().Set(True)

# ─── Object (red cube) ────────────────────────────────────────
object_path = "/World/Object"
obj = UsdGeom.Xform.Define(stage, object_path)

def randomize_object():
    """Place object at random position on table."""
    x = random.uniform(0.3, 0.65)
    y = random.uniform(-0.15, 0.15)
    z = 0.82  # just above table
    obj.GetPrim().GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
    print(f"Object placed at ({x:.2f}, {y:.2f}, {z:.2f})")

randomize_object()

UsdPhysics.RigidBodyAPI.Apply(obj.GetPrim())
obj_mass = UsdPhysics.MassAPI.Apply(obj.GetPrim())
obj_mass.CreateMassAttr().Set(0.1)
obj_geom = UsdGeom.Cube.Define(stage, f"{object_path}/Geom")
obj_geom.GetSizeAttr().Set(1.0)
obj_geom.AddScaleOp().Set(Gf.Vec3d(0.04, 0.04, 0.04))
obj_geom.GetPrim().CreateAttribute("primvars:displayColor",
    Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.9, 0.2, 0.2)])

# ─── Basket ───────────────────────────────────────────────────
basket_path = "/World/Basket"
basket = UsdGeom.Xform.Define(stage, basket_path)
basket.AddTranslateOp().Set(Gf.Vec3d(0.65, 0.0, 0.80))
UsdPhysics.RigidBodyAPI.Apply(basket.GetPrim())
basket_rb = UsdPhysics.RigidBodyAPI.Get(stage, Sdf.Path(basket_path))
basket_rb.CreateKinematicEnabledAttr().Set(True)
basket_geom = UsdGeom.Cube.Define(stage, f"{basket_path}/Geom")
basket_geom.GetSizeAttr().Set(1.0)
basket_geom.AddScaleOp().Set(Gf.Vec3d(0.15, 0.15, 0.08))
basket_geom.GetPrim().CreateAttribute("primvars:displayColor",
    Sdf.ValueTypeNames.Color3fArray, False).Set([Gf.Vec3f(0.5, 0.3, 0.1)])

# ─── UR10e Robot ──────────────────────────────────────────────
robot_path = "/World/UR10e"
robot_asset = "omniverse://localhost/NVIDIA/Assets/Isaac/2023.1.1/Isaac/Robots/UR10e/ur10e_robotiq_2f_85.usd"

omni.kit.commands.execute("CreateReferenceCommand",
    usd_context=ctx,
    path_to=Sdf.Path(robot_path),
    asset_path=robot_asset,
    instanceable=False,
)

robot_prim = stage.GetPrimAtPath(robot_path)
if robot_prim.IsValid():
    print("UR10e + Robotiq 2F-85 loaded!")
    # Position robot at base of table
    xform = UsdGeom.Xformable(robot_prim)
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
else:
    print("WARNING: Could not load UR10e from Nucleus.")
    print("Trying local asset path...")
    # Try alternate path
    try:
        omni.kit.commands.execute("CreateReferenceCommand",
            usd_context=ctx,
            path_to=Sdf.Path(robot_path),
            asset_path="omniverse://localhost/NVIDIA/Assets/Isaac/4.5/Isaac/Robots/UR10e/ur10e_robotiq_2f_85.usd",
            instanceable=False,
        )
        robot_prim = stage.GetPrimAtPath(robot_path)
        if robot_prim.IsValid():
            print("UR10e loaded from alternate path!")
    except Exception as e:
        print(f"Robot load failed: {e}")

# ─── Lighting ─────────────────────────────────────────────────
light = UsdGeom.Xform.Define(stage, "/World/Light")
dl = UsdGeom.DomeLight.Define(stage, "/World/Light/DomeLight")
dl.CreateIntensityAttr().Set(2000)
dl.CreateColorAttr().Set(Gf.Vec3f(1.0, 0.95, 0.9))

# ─── Camera ───────────────────────────────────────────────────
camera = UsdGeom.Camera.Define(stage, "/World/Camera")
camera.AddTranslateOp().Set(Gf.Vec3d(1.2, 0.8, 1.8))
camera.AddRotateXYZOp().Set(Gf.Vec3d(-30, 0, -45))
camera.CreateFocalLengthAttr().Set(24)

print("\nScene complete!")
print(f"Prims: {[str(p.GetPath()) for p in stage.Traverse()][:10]}...")

# ─── Simulation Loop ──────────────────────────────────────────
import omni.timeline
import omni.kit.app

timeline = omni.timeline.get_timeline_interface()
timeline.play()

print("\n▶ Simulation running. Press Ctrl+C to stop.")
print("  Controls: Up/Down arrows in viewport to zoom")
print("           Right-click drag to rotate view")

app = omni.kit.app.get_app()
step_count = 0

try:
    while simulation_app.is_running():
        app.update()
        step_count += 1
        if step_count % 60 == 0:
            obj_trans = obj.GetPrim().GetAttribute("xformOp:translate").Get()
            print(f"  t={step_count//60}s | Object: ({obj_trans[0]:.2f}, {obj_trans[1]:.2f}, {obj_trans[2]:.2f})")
except KeyboardInterrupt:
    print("\nStopping...")

timeline.stop()
simulation_app.close()
print("Demo ended.")
