# 6-DOF arm pick-and-place

An Isaac Lab project for reaching and grasping objects with a UR10e arm and a
Robotiq 2F-85 gripper.

The current controller is hybrid: PPO moves the wrist to a target-conditioned
pre-grasp pose, then a deterministic 6-DOF pose servo handles descent, gripper
closure, lift, and one corrective retry. This keeps contact behavior observable
and avoids training an end-to-end manipulation policy before the simulator setup
is stable.

## Project status

| Stage | Verified result | Status |
|---|---:|---|
| Phase 0: reach | 256/256 within 5 cm; 7 mm median closest distance | Complete |
| Phase 1: hybrid grasp | 78/128 strict lifts (60.9%) across two seeds | In progress |
| Four-arm fixed layout | 4/4 reached; 2/4 lifted | Visual smoke test |
| Phase 2: place | Not implemented | Blocked on reliable grasping |

The Phase 1 benchmark uses a 4 x 4 x 10 cm upright block. The original 4 cm
cube is not yet reliable because its usable finger-contact band is much smaller.
The current Phase 1 exit target is at least 80% strict lift success with low
integrity-reset rates and a stable two-second hold.

Model files are intentionally excluded from Git because they are binary
artifacts. The result paths used during development were:

- `models/isaac/phase_0/model_pregrasp_coupled_v2.pt`
- `models/isaac/phase_1/model_grasp_v1.pt`

A fresh clone does not contain these checkpoints. Train them locally or copy
compatible checkpoints into those paths before running the evaluation commands.

## How it works

```text
object XYZ + robot state -> learned pre-grasp policy
                                  |
                                  v
                    pose servo -> close -> lift
                                      |
                                      +-> recenter and retry once if needed
```

The trained policy currently receives structured state, not video. Its 34-value
observation contains:

| Input | Values |
|---|---:|
| Six arm joint positions and velocities | 12 |
| Wrist position and quaternion | 7 |
| Gripper state | 1 |
| Object position, relative vector, and distance | 7 |
| Previous action | 7 |
| **Total** | **34** |

For a future physical setup, a camera detector will replace the simulator's
object-position source. Robot joint feedback will still come from the robot and
remain part of closed-loop control. The repository already contains a target
tracker that rejects stale, low-confidence, non-finite, out-of-workspace, and
implausibly jumping detections.

Camera-position perturbation tests produced these Phase 0 results:

| Synthetic calibration error | Reach success within 5 cm |
|---|---:|
| Static +3 cm on X and +3 cm on Y | 64/64 |
| Static +5 cm on X and +5 cm on Y | 19/64 |

This suggests that 3 cm is an outer reach-only bound. Grasping a 4 cm-wide
object will require substantially better calibration; the current engineering
target is approximately 1 cm persistent position error.

## Repository layout

```text
isaac_env/
  actions.py           bounded arm and slew-limited gripper actions
  env_cfg.py           Isaac Lab scene and manager configuration
  mdp.py               observations, rewards, resets, and terminations
  target_provider.py   camera-independent target validation and smoothing
scripts/
  train_isaac.py       PPO training and warm-start entry point
  evaluate_isaac.py    evaluation, ablations, and hybrid grasp demo
spikes/                recorded diagnostic experiments and conclusions
tests/                 hardware-independent smoke tests
PLAN.md                ordered roadmap and completion gates
```

## Setup

The current results were produced on:

- Windows 11
- Python 3.11.9
- Isaac Sim 5.1.0
- PyTorch 2.7.0 with CUDA 12.8
- NVIDIA RTX 3060 Laptop GPU with 6 GB VRAM

Isaac Sim, Isaac Lab, and RSL-RL are external dependencies and are not installed
by this repository's `requirements.txt`. Set up an Isaac Lab environment first.
The project was tested with the official
[Isaac Sim 5.1 pip installation workflow](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/pip_installation.html)
and Python 3.11.

After activating that environment:

```powershell
git clone https://github.com/sting-raider/6DOF-arm-RL.git
cd 6DOF-arm-RL
python -m pip install -r requirements.txt
```

The local development checkout uses `.venv\Scripts\python.exe`. Substitute the
Python executable from your Isaac Lab environment if yours has a different name.

## Train

Phase 0 training:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe scripts\train_isaac.py `
  --phase 0 --num_envs 512 --max_iterations 1000 --headless `
  --output_model models\isaac\phase_0\model.pt
```

Warm-starting transfers the actor and its observation normalizer while leaving
the Phase 1 critic and optimizer fresh:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe scripts\train_isaac.py `
  --phase 1 --num_envs 512 --max_iterations 100 --headless `
  --warm_start models\isaac\phase_0\model.pt `
  --output_model models\isaac\phase_1\model.pt
```

Phase 0 does not need to be retrained for the current development checkpoint.
Use the cloud GPU only after local deterministic control reaches a measured
plateau; the roadmap limits any cloud run to a small residual correction policy.

## Evaluate

The following commands assume the development checkpoints exist at the paths
shown near the top of this README.

Phase 0 regression:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 0 `
  --model models\isaac\phase_0\model_pregrasp_coupled_v2.pt `
  --num_envs 256 --episodes 256 --headless `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

Hybrid Phase 1 benchmark:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 1 `
  --model models\isaac\phase_1\model_grasp_v1.pt `
  --num_envs 64 --episodes 64 --seed 42 `
  --hybrid_phase1 --headless `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

Visible four-arm demo:

```powershell
$env:OMNI_KIT_ACCEPT_EULA='YES'
.\.venv\Scripts\python.exe -u scripts\evaluate_isaac.py `
  --phase 1 `
  --model models\isaac\phase_1\model_grasp_v1.pt `
  --num_envs 4 --episodes 4 --seed 42 `
  --hybrid_phase1 --demo_layout --realtime `
  --kit_args "--/app/vulkan=false --/renderer/multiGpu/enabled=false --/renderer/multiGpu/autoEnable=false"
```

The D3D renderer flags above are required by the tested Windows laptop setup.
They may not be appropriate on Linux or a different GPU configuration.

## Tests

The fast suite does not launch Isaac Sim:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q
```

It checks repository structure, Python syntax, configuration parsing, and target
tracker behavior. Contact dynamics still require an Isaac evaluation run.

## Known limitations

- Phase 1 succeeds on 60.9% of the current starter-object benchmark, not at a
  production-ready rate.
- Gripper and arm integrity resets remain the main simulator failure mode.
- The raw RGB detector and camera calibration pipeline are not connected yet.
- The 4 cm cube and varied object shapes are not reliable.
- Transport, release, and basket placement are not implemented.
- Simulation results do not demonstrate real-world safety or transfer.

See [PLAN.md](PLAN.md) for the ordered milestones and completion criteria. The
diagnostic evidence behind the current decisions is recorded under
[`spikes/`](spikes/).

## License

Released under the [MIT License](LICENSE).
