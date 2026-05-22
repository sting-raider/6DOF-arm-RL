"""
Isaac Sim 5.1.0 GPU verification test.
Run inside container: /isaac-sim/python.sh /workspace/scripts/test_isaac.py
"""
from isaacsim import SimulationApp
print('STEP 1: Import OK')

simulation_app = SimulationApp({
    "headless": True,
    "renderer": "RayTracedLighting",
})

print('STEP 2: SimulationApp started (headless)')

import omni.usd
from pxr import Usd, UsdGeom, Gf

stage = Usd.Stage.CreateInMemory()
omni.usd.get_context().set_stage(stage)
print('STEP 3: USD stage created')

cube = UsdGeom.Cube.Define(stage, '/World/TestCube')
cube.GetSizeAttr().Set(0.5)
cube.AddTranslateOp().Set(Gf.Vec3d(0, 0, 1))
print('STEP 4: TestCube at height 1.0')

print('=== ALL STEPS PASSED — Isaac Sim 5.1.0 + GPU ===')
print('Stage prims:', [str(p.GetPath()) for p in stage.Traverse()])

simulation_app.close()
print('Test complete.')
