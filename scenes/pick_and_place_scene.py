"""
MuJoCo scene for the 6-DOF arm pick-and-place task.

This file defines the complete physics world with:
  - A 6-DOF robotic arm with parallel-jaw gripper
  - A static table
  - A static basket (open-top box)
  - A dynamic graspable cube
  - Camera sensor attached to a fixed frame
"""

import os

# Paths are relative to the scenes/ directory where this script resides.
# Users should call build_scene_xml() and write to a file.

SCENE_XML = """<mujoco model="pick_and_place">

  <!-- Physics & solver settings -->
  <option timestep="0.002" integrator="RK4" gravity="0 0 -9.81" solver="Newton"
          noslip_iterations="5" noslip_tolerance="1e-10"
          contact="all" cone="elliptic"/>

  <size nconmax="100" njmax="100"/>

  <!-- Default inertial and contact properties -->
  <default>
    <geom rgba="0.5 0.5 0.5 1" density="800" contyp="6" conaffinity="6" />
    <joint type="hinge" limited="true" damping="0.5" armature="0.01" frictionloss="0.05"/>
    <joint type="slide" limited="true" damping="0.5" armature="0.01" frictionloss="0.05"/>
    <motor ctrllimited="true" ctrlrange="-1 1" gear="1"/>
    <site rgba="1 0 0 1" size="0.01"/>
    <general ctrllimited="true" ctrlrange="-1 1" gear="1" biastype="affine" gaintype="affine" dynprm="0 1 1"/>
  </default>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.9 0.9 0.9" rgb2="0.5 0.5 0.5" width="256" height="256"/>
    <texture name="table_tex" type="2d" builtin="checker" rgb1="0.7 0.7 0.7" rgb2="0.3 0.3 0.3" width="256" height="256"/>
    <material name="table_mat" texture="table_tex" shininess="0.1"/>
    <material name="basket_mat" rgba="0.6 0.3 0.1 1" shininess="0.5"/>
    <material name="arm_mat" rgba="0.3 0.5 0.7 1" shininess="0.5"/>
    <material name="gripper_mat" rgba="0.5 0.5 0.5 1" shininess="0.8"/>
    <material name="object_mat" rgba="0.8 0.3 0.2 1" shininess="0.6"/>
    <material name="ground_mat" rgba="0.2 0.2 0.2 1"/>
  </asset>

  <worldbody>
    <!-- Ground plane -->
    <geom name="ground" type="plane" size="5 5 0.01" material="ground_mat" contype="2" conaffinity="2"/>

    <!-- Table -->
    <body name="table" pos="0 0 0.79">
      <geom name="table_surface" type="box" size="0.3 0.2 0.01" material="table_mat"
            contype="2" conaffinity="2" friction="1.0 0.1 0.1"/>
      <geom name="table_leg1" type="cylinder" size="0.02 0.4" pos="-0.25 -0.15 -0.4"/>
      <geom name="table_leg2" type="cylinder" size="0.02 0.4" pos="0.25 -0.15 -0.4"/>
      <geom name="table_leg3" type="cylinder" size="0.02 0.4" pos="-0.25 0.15 -0.4"/>
      <geom name="table_leg4" type="cylinder" size="0.02 0.4" pos="0.25 0.15 -0.4"/>
    </body>

    <!-- Basket (open-top box constructed from walls) -->
    <body name="basket" pos="0.4 0 0.8">
      <!-- Bottom -->
      <geom name="basket_bottom" type="box" size="0.075 0.075 0.005" material="basket_mat"
            pos="0 0 -0.005" friction="0.8 0.1 0.1" contype="2" conaffinity="2"/>
      <!-- Walls -->
      <geom name="basket_wall1" type="box" size="0.075 0.005 0.05" material="basket_mat"
            pos="0 0.08 0.045" friction="0.8 0.1 0.1" contype="2" conaffinity="2"/>
      <geom name="basket_wall2" type="box" size="0.075 0.005 0.05" material="basket_mat"
            pos="0 -0.08 0.045" friction="0.8 0.1 0.1" contype="2" conaffinity="2"/>
      <geom name="basket_wall3" type="box" size="0.005 0.075 0.05" material="basket_mat"
            pos="0.08 0 0.045" friction="0.8 0.1 0.1" contype="2" conaffinity="2"/>
      <geom name="basket_wall4" type="box" size="0.005 0.075 0.05" material="basket_mat"
            pos="-0.08 0 0.045" friction="0.8 0.1 0.1" contype="2" conaffinity="2"/>
    </body>

    <!-- Graspable Object ( Cube ) -->
    <body name="object" pos="0 0 0.83">
      <freejoint name="object_free"/>
      <inertial pos="0 0 0" mass="0.1" diaginertia="0.0001 0.0001 0.0001"/>
      <geom name="object_geom" type="box" size="0.02 0.02 0.02" material="object_mat"
            contype="2" conaffinity="2" friction="0.8 0.1 0.1"/>
    </body>

    <!-- 6-DOF KUKA iiwa Arm -->
    <!-- Base is fixed at origin, slightly above ground for table clearance -->
    <body name="kuka_base" pos="-0.3 0 0.02">
      <!-- Base link -->
      <geom name="base_geom" type="cylinder" size="0.04 0.02" material="arm_mat" contype="1" conaffinity="1"/>
      
      <!-- Joint 1: Base rotation (vertical) -->
      <body name="link1" pos="0 0 0.02">
        <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.5"/>
        <geom name="link1_geom" type="capsule" size="0.03 0.13" pos="0 0 0.085" material="arm_mat" contype="1" conaffinity="1"/>
        
        <!-- Joint 2: Shoulder -->
        <body name="link2" pos="0 0 0.17">
          <joint name="joint2" type="hinge" axis="0 -1 0" range="-1.57 1.57" damping="0.5"/>
          <geom name="link2_geom" type="capsule" size="0.03 0.13" pos="0 0 0.085" material="arm_mat" contype="1" conaffinity="1"/>
          
          <!-- Joint 3: Elbow -->
          <body name="link3" pos="0 0 0.17">
            <joint name="joint3" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.5"/>
            <geom name="link3_geom" type="capsule" size="0.028 0.12" pos="0 0 0.08" material="arm_mat" contype="1" conaffinity="1"/>
            
            <!-- Joint 4: Wrist 1 -->
            <body name="link4" pos="0 0 0.16">
              <joint name="joint4" type="hinge" axis="0 1 0" range="-3.14 3.14" damping="0.3"/>
              <geom name="link4_geom" type="capsule" size="0.025 0.1" pos="0 0 0.065" material="arm_mat" contype="1" conaffinity="1"/>
              
              <!-- Joint 5: Wrist 2 -->
              <body name="link5" pos="0 0 0.13">
                <joint name="joint5" type="hinge" axis="0 -1 0" range="-1.57 1.57" damping="0.3"/>
                <geom name="link5_geom" type="capsule" size="0.022 0.07" pos="0 0 0.05" material="arm_mat" contype="1" conaffinity="1"/>
                
                <!-- Joint 6: Wrist 3 -->
                <body name="link6" pos="0 0 0.1">
                  <joint name="joint6" type="hinge" axis="0 0 1" range="-3.14 3.14" damping="0.3"/>
                  <geom name="link6_geom" type="capsule" size="0.02 0.05" pos="0 0 0.035" material="arm_mat" contype="1" conaffinity="1"/>
                  
                  <!-- End effector mount -->
                  <body name="ee_mount" pos="0 0 0.07">
                    <!-- Parallel-Jaw Gripper -->
                    <!-- Left finger -->
                    <body name="gripper_left" pos="0 -0.025 0">
                      <joint name="gripper_left_joint" type="slide" axis="0 -1 0" range="-0.02 0.02" damping="0.5"/>
                      <geom name="gripper_left_geom" type="box" size="0.015 0.005 0.03" material="gripper_mat" contype="1" conaffinity="1"/>
                    </body>
                    <!-- Right finger -->
                    <body name="gripper_right" pos="0 0.025 0">
                      <joint name="gripper_right_joint" type="slide" axis="0 1 0" range="-0.02 0.02" damping="0.5"/>
                      <geom name="gripper_right_geom" type="box" size="0.015 0.005 0.03" material="gripper_mat" contype="1" conaffinity="1"/>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Overhead camera (for vision simulation) -->
    <body name="camera_frame" pos="0 0 2.0">
      <!-- Camera geom for visualisation -->
      <geom name="camera_housing" type="box" size="0.01 0.01 0.01" rgba="0 0 1 1"/>
      <camera name="overhead" mode="fixed" pos="0 0 0" euler="-1.5708 0 0" fovy="45"/>
    </body>
  </worldbody>

  <!-- Actuators -->
  <actuator>
    <motor joint="joint1" name="actuator_joint1" gear="50"/>
    <motor joint="joint2" name="actuator_joint2" gear="50"/>
    <motor joint="joint3" name="actuator_joint3" gear="50"/>
    <motor joint="joint4" name="actuator_joint4" gear="40"/>
    <motor joint="joint5" name="actuator_joint5" gear="40"/>
    <motor joint="joint6" name="actuator_joint6" gear="40"/>
    <motor joint="gripper_left_joint" name="actuator_gripper_left" gear="10"/>
    <motor joint="gripper_right_joint" name="actuator_gripper_right" gear="10"/>
  </actuator>

  <!-- Equality constraints (for magnetic grasping) -->
  <equality>
    <!-- Weld between object and end-effector when grasp is active -->
    <weld body1="kuka_base" body2="object" anchor="0 0 0" solref="0.02 1" solimp="0.9 0.95 0.001" active="0" name="grasp_weld"/>
  </equality>

  <!-- Sensors -->
  <sensor>
    <!-- End-effector position -->
    <framepos name="ee_pos" objtype="xbody" objname="link6"/>
  </sensor>

</mujoco>
"""

def build_scene_xml() -> str:
    """Return the full MuJoCo XML string."""
    return SCENE_XML


def write_scene_xml(path: str) -> None:
    """Write the scene XML to a file."""
    with open(path, 'w') as f:
        f.write(SCENE_XML)
    print(f"Scene XML written to {path}")


if __name__ == "__main__":
    out_path = os.path.join(os.path.dirname(__file__), "pick_and_place_scene.xml")
    write_scene_xml(out_path)
