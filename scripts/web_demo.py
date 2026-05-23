#!/usr/bin/env python3
"""
Web-based evaluation dashboard for the 6-DOF arm SAC agent.

Serves a live dark-theme dashboard at http://localhost:8080 showing:
  - Live metrics: current episode reward, steps, success flags
  - Last-10 episode history table
  - Auto-refreshes every 2 seconds

Observation space (20D):
  ee_pos(3) + obj_pos(3) + relative_pos(3) +
  joint_pos(5) + joint_vel(5) + gripper_state(1)

Usage:
    python scripts/web_demo.py                          # default: models/phase_2/best_model
    python scripts/web_demo.py --model models/phase_1/best_model
    python scripts/web_demo.py --phase 1
    python scripts/web_demo.py --port 8080
"""

import sys
import os
import argparse
import threading
import time
import json
import pickle
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — make sure project root is importable
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from envs.pick_and_place_env import PickAndPlaceEnv
from utils.constants import TABLE_HEIGHT

# ---------------------------------------------------------------------------
# Shared state — all writes/reads protected by STATE_LOCK
# ---------------------------------------------------------------------------
STATE_LOCK = threading.Lock()

# Live (current episode) state
LIVE_STATE = {
    "episode": 0,
    "step": 0,
    "reward": 0.0,
    "cumulative_reward": 0.0,
    "reach_success": False,
    "grasp_success": False,
    "place_success": False,
    "ee_pos": [0.0, 0.0, 0.0],
    "obj_pos": [0.0, 0.0, 0.0],
    "gripper": 0.0,
    "model_path": "",
    "status": "starting",          # "starting" | "running" | "error"
    "error_msg": "",
}

# Rolling history of last 10 completed episodes
EPISODE_HISTORY: deque = deque(maxlen=10)

# ---------------------------------------------------------------------------
# HTML template (dark theme, auto-refresh every 2 s)
# ---------------------------------------------------------------------------
HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>6-DOF Arm RL Dashboard</title>
  <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
      --bg:       #0d1117;
      --surface:  #161b22;
      --surface2: #21262d;
      --border:   #30363d;
      --accent:   #58a6ff;
      --green:    #3fb950;
      --red:      #f85149;
      --yellow:   #d29922;
      --purple:   #bc8cff;
      --text:     #e6edf3;
      --muted:    #8b949e;
      --radius:   10px;
    }

    * { margin: 0; padding: 0; box-sizing: border-box; }

    body {
      font-family: 'Inter', sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      padding: 24px;
    }

    header {
      display: flex;
      align-items: center;
      gap: 14px;
      margin-bottom: 28px;
    }

    header .logo { font-size: 32px; }

    header h1 {
      font-size: 22px;
      font-weight: 700;
      color: var(--accent);
    }

    header .sub {
      font-size: 12px;
      color: var(--muted);
      margin-top: 2px;
      font-family: 'JetBrains Mono', monospace;
    }

    .pill {
      padding: 3px 10px;
      border-radius: 20px;
      font-size: 11px;
      font-weight: 600;
      letter-spacing: 0.5px;
      text-transform: uppercase;
    }

    .pill.running  { background: rgba(63,185,80,0.15); color: var(--green); border: 1px solid rgba(63,185,80,0.4); }
    .pill.starting { background: rgba(210,153,34,0.15); color: var(--yellow); border: 1px solid rgba(210,153,34,0.4); }
    .pill.error    { background: rgba(248,81,73,0.15);  color: var(--red);   border: 1px solid rgba(248,81,73,0.4); }

    /* ---- KPI cards ---- */
    .kpi-grid {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 24px;
    }

    .kpi {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      padding: 18px 20px;
      transition: border-color 0.3s;
    }

    .kpi:hover { border-color: var(--accent); }

    .kpi .label {
      font-size: 11px;
      color: var(--muted);
      font-weight: 500;
      text-transform: uppercase;
      letter-spacing: 0.6px;
      margin-bottom: 8px;
    }

    .kpi .value {
      font-family: 'JetBrains Mono', monospace;
      font-size: 26px;
      font-weight: 700;
      color: var(--accent);
      line-height: 1;
    }

    .kpi .value.success { color: var(--green); }
    .kpi .value.fail    { color: var(--red); }
    .kpi .value.neutral { color: var(--yellow); }

    /* ---- Sections ---- */
    .section {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: var(--radius);
      margin-bottom: 20px;
      overflow: hidden;
    }

    .section-header {
      padding: 14px 20px;
      font-size: 13px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.7px;
      border-bottom: 1px solid var(--border);
      background: var(--surface2);
    }

    .section-body { padding: 18px 20px; }

    /* ---- Obs grid ---- */
    .obs-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
    }

    .obs-item .obs-label {
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 4px;
    }

    .obs-item .obs-val {
      font-family: 'JetBrains Mono', monospace;
      font-size: 13px;
      color: var(--text);
    }

    /* ---- Success badges ---- */
    .badges { display: flex; gap: 10px; flex-wrap: wrap; }

    .badge {
      padding: 5px 14px;
      border-radius: 20px;
      font-size: 12px;
      font-weight: 600;
      border: 1.5px solid;
      transition: all 0.3s;
    }

    .badge.on  { background: rgba(63,185,80,0.15); color: var(--green); border-color: var(--green); }
    .badge.off { background: transparent; color: var(--muted); border-color: var(--border); }

    /* ---- Progress bar ---- */
    .progress-wrap {
      height: 6px;
      background: var(--surface2);
      border-radius: 3px;
      overflow: hidden;
      margin-top: 10px;
    }

    .progress-bar {
      height: 100%;
      background: linear-gradient(90deg, var(--accent), var(--purple));
      border-radius: 3px;
      transition: width 0.4s ease;
    }

    /* ---- History table ---- */
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }

    thead th {
      padding: 9px 12px;
      text-align: left;
      font-size: 11px;
      font-weight: 600;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      border-bottom: 1px solid var(--border);
    }

    tbody tr {
      border-bottom: 1px solid rgba(48,54,61,0.6);
      transition: background 0.15s;
    }

    tbody tr:hover { background: var(--surface2); }

    tbody td {
      padding: 9px 12px;
      font-family: 'JetBrains Mono', monospace;
      color: var(--text);
    }

    td.ok   { color: var(--green); }
    td.fail { color: var(--muted); }

    /* ---- Footer ---- */
    footer {
      margin-top: 24px;
      text-align: center;
      font-size: 11px;
      color: var(--muted);
    }

    #last-update { color: var(--accent); }

    .spinner {
      display: inline-block;
      width: 8px; height: 8px;
      border: 2px solid var(--muted);
      border-top-color: var(--accent);
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
      margin-right: 5px;
      vertical-align: middle;
    }

    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>

<header>
  <span class="logo">🦾</span>
  <div>
    <h1>6-DOF Arm RL Dashboard</h1>
    <div class="sub" id="model-path">Loading...</div>
  </div>
  <div style="margin-left: auto; display:flex; align-items:center; gap:10px;">
    <span class="spinner"></span>
    <span id="status-pill" class="pill starting">Starting</span>
  </div>
</header>

<!-- KPI Cards -->
<div class="kpi-grid">
  <div class="kpi">
    <div class="label">Episode</div>
    <div class="value" id="kpi-ep">—</div>
  </div>
  <div class="kpi">
    <div class="label">Step</div>
    <div class="value" id="kpi-step">—</div>
  </div>
  <div class="kpi">
    <div class="label">Ep. Reward</div>
    <div class="value" id="kpi-reward">—</div>
  </div>
  <div class="kpi">
    <div class="label">Avg Reward (last 10)</div>
    <div class="value" id="kpi-avg">—</div>
  </div>
  <div class="kpi">
    <div class="label">Place Success %</div>
    <div class="value" id="kpi-place">—</div>
  </div>
</div>

<!-- Live obs -->
<div class="section">
  <div class="section-header">Live Observation</div>
  <div class="section-body">
    <div class="obs-grid">
      <div class="obs-item">
        <div class="obs-label">End-Effector Position</div>
        <div class="obs-val" id="obs-ee">—</div>
      </div>
      <div class="obs-item">
        <div class="obs-label">Object Position</div>
        <div class="obs-val" id="obs-obj">—</div>
      </div>
      <div class="obs-item">
        <div class="obs-label">Gripper State</div>
        <div class="obs-val" id="obs-grip">—</div>
      </div>
    </div>
    <div style="margin-top:14px;">
      <div class="obs-label">Episode progress</div>
      <div class="progress-wrap"><div class="progress-bar" id="ep-bar" style="width:0%"></div></div>
    </div>
  </div>
</div>

<!-- Success badges -->
<div class="section">
  <div class="section-header">Current Episode — Task Progress</div>
  <div class="section-body">
    <div class="badges">
      <div class="badge off" id="badge-reach">✓ Reach</div>
      <div class="badge off" id="badge-grasp">✓ Grasp</div>
      <div class="badge off" id="badge-place">✓ Place</div>
    </div>
  </div>
</div>

<!-- History table -->
<div class="section">
  <div class="section-header">Last 10 Episodes</div>
  <div class="section-body" style="padding: 0;">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Steps</th>
          <th>Total Reward</th>
          <th>Reach</th>
          <th>Grasp</th>
          <th>Place</th>
        </tr>
      </thead>
      <tbody id="history-body">
        <tr><td colspan="6" style="text-align:center; color:var(--muted); padding:20px;">Waiting for first episode...</td></tr>
      </tbody>
    </table>
  </div>
</div>

<footer>
  Auto-refresh every 2 s &nbsp;|&nbsp; Last update: <span id="last-update">—</span>
</footer>

<script>
  const MAX_EP_STEPS = 400; // phase 2; adjust if needed

  async function refresh() {
    try {
      const r = await fetch('/api/state');
      const d = await r.json();

      // Header
      document.getElementById('model-path').textContent = d.model_path || 'No model loaded';
      const pill = document.getElementById('status-pill');
      pill.textContent = d.status;
      pill.className = 'pill ' + d.status;

      // KPIs
      document.getElementById('kpi-ep').textContent     = d.episode;
      document.getElementById('kpi-step').textContent   = d.step;
      document.getElementById('kpi-reward').textContent = d.cumulative_reward.toFixed(2);

      const hist = d.history;
      let avgR = '—', placeRate = '—';
      if (hist.length > 0) {
        avgR = (hist.reduce((a, e) => a + e.reward, 0) / hist.length).toFixed(2);
        placeRate = ((hist.filter(e => e.place_success).length / hist.length) * 100).toFixed(0) + '%';
      }
      document.getElementById('kpi-avg').textContent   = avgR;
      document.getElementById('kpi-place').textContent = placeRate;

      // Obs
      const f3 = v => v.map(x => x.toFixed(3)).join(', ');
      document.getElementById('obs-ee').textContent   = '[' + f3(d.ee_pos) + ']';
      document.getElementById('obs-obj').textContent  = '[' + f3(d.obj_pos) + ']';
      document.getElementById('obs-grip').textContent = d.gripper.toFixed(3);

      // Progress bar
      const pct = Math.min(100, (d.step / MAX_EP_STEPS) * 100);
      document.getElementById('ep-bar').style.width = pct + '%';

      // Badges
      setBadge('badge-reach', d.reach_success);
      setBadge('badge-grasp', d.grasp_success);
      setBadge('badge-place', d.place_success);

      // History table
      const tbody = document.getElementById('history-body');
      if (hist.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center; color:var(--muted); padding:20px;">Waiting for first episode...</td></tr>';
      } else {
        tbody.innerHTML = hist.slice().reverse().map(ep => `
          <tr>
            <td>${ep.episode}</td>
            <td>${ep.steps}</td>
            <td>${ep.reward.toFixed(2)}</td>
            <td class="${ep.reach_success ? 'ok' : 'fail'}">${ep.reach_success ? '✓' : '✗'}</td>
            <td class="${ep.grasp_success ? 'ok' : 'fail'}">${ep.grasp_success ? '✓' : '✗'}</td>
            <td class="${ep.place_success ? 'ok' : 'fail'}">${ep.place_success ? '✓' : '✗'}</td>
          </tr>`).join('');
      }

      document.getElementById('last-update').textContent = new Date().toLocaleTimeString();
    } catch(e) {
      console.error('Refresh failed:', e);
    }
  }

  function setBadge(id, active) {
    const el = document.getElementById(id);
    el.className = 'badge ' + (active ? 'on' : 'off');
  }

  refresh();
  setInterval(refresh, 2000);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP request handler (stdlib only — no Flask)
# ---------------------------------------------------------------------------
class DashboardHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler serving the dashboard and the JSON API."""

    def log_message(self, fmt, *args):
        # Suppress noisy access logs (only errors go to stderr via log_error)
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html()
        elif self.path == "/api/state":
            self._send_state()
        else:
            self.send_response(404)
            self.end_headers()

    def _send_html(self):
        data = HTML_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_state(self):
        with STATE_LOCK:
            payload = {
                **LIVE_STATE,
                "history": list(EPISODE_HISTORY),
            }
        data = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)


# ---------------------------------------------------------------------------
# Evaluation loop (runs in background thread)
# ---------------------------------------------------------------------------

def find_model_and_vecnorm(phase: int, model_path_override: str | None):
    """
    Resolve model zip path and VecNormalize pkl path.

    Search priority (model):
      1. --model CLI flag (exact path, with or without .zip)
      2. models/phase_{phase}/best_model
      3. models/phase_{phase}/final_model
      4. Any .zip in models/phase_{phase}/

    VecNormalize: always models/phase_{phase}/vec_normalize.pkl
    """
    phase_dir = PROJECT_DIR / "models" / f"phase_{phase}"

    # --- Model ---
    if model_path_override:
        p = Path(model_path_override)
        if not p.is_absolute():
            p = PROJECT_DIR / p
        # SB3 SAC.load accepts path without .zip
        if not p.suffix:
            p = p.with_suffix(".zip")
        if not p.exists():
            raise FileNotFoundError(f"Model not found: {p}")
        model_path = p.with_suffix("")  # SAC.load without extension
    else:
        candidates = [
            phase_dir / "best_model.zip",
            phase_dir / "final_model.zip",
        ]
        model_path = None
        for c in candidates:
            if c.exists():
                model_path = c.with_suffix("")
                break
        if model_path is None:
            zips = list(phase_dir.glob("*.zip"))
            if zips:
                model_path = sorted(zips)[-1].with_suffix("")
        if model_path is None:
            raise FileNotFoundError(
                f"No model .zip found in {phase_dir}. "
                "Train one first or pass --model <path>."
            )

    # --- VecNormalize ---
    vec_pkl = phase_dir / "vec_normalize.pkl"

    return model_path, vec_pkl


def make_env(phase: int, xml_path: str):
    """Factory for PickAndPlaceEnv wrapped in DummyVecEnv."""
    def _make():
        return PickAndPlaceEnv(
            xml_path=xml_path,
            curriculum_phase=phase,
            use_vision=False,
            render_mode="rgb_array",
        )
    return DummyVecEnv([_make])


def eval_loop(phase: int, model_path_override: str | None, num_episodes: int):
    """
    Background thread: loads model + VecNormalize, runs episodes forever
    (or until num_episodes if finite), updating LIVE_STATE after each step.
    """
    xml_path = str(PROJECT_DIR / "scenes" / "pick_and_place_scene.xml")

    try:
        model_path, vec_pkl = find_model_and_vecnorm(phase, model_path_override)
    except FileNotFoundError as exc:
        with STATE_LOCK:
            LIVE_STATE["status"] = "error"
            LIVE_STATE["error_msg"] = str(exc)
        print(f"[web_demo] ERROR: {exc}")
        return

    # Update status: loading
    with STATE_LOCK:
        LIVE_STATE["model_path"] = str(model_path) + ".zip"
        LIVE_STATE["status"] = "starting"

    print(f"[web_demo] Loading model: {model_path}.zip")
    try:
        model = SAC.load(str(model_path))
    except Exception as exc:
        with STATE_LOCK:
            LIVE_STATE["status"] = "error"
            LIVE_STATE["error_msg"] = f"SAC.load failed: {exc}"
        print(f"[web_demo] ERROR loading model: {exc}")
        return

    # Build env
    print(f"[web_demo] Building environment (phase {phase}) ...")
    try:
        vec_env = make_env(phase, xml_path)
    except Exception as exc:
        with STATE_LOCK:
            LIVE_STATE["status"] = "error"
            LIVE_STATE["error_msg"] = f"Env init failed: {exc}"
        print(f"[web_demo] ERROR building env: {exc}")
        return

    # Optionally wrap with VecNormalize
    if vec_pkl.exists():
        print(f"[web_demo] Loading VecNormalize stats: {vec_pkl}")
        try:
            with open(vec_pkl, "rb") as f:
                saved_stats = pickle.load(f)
            vec_env = VecNormalize.load(str(vec_pkl), vec_env)
            vec_env.training = False        # freeze stats during eval
            vec_env.norm_reward = False     # don't normalize rewards for display
        except Exception as exc:
            print(f"[web_demo] WARNING: Could not load VecNormalize: {exc}. Running without.")
    else:
        print(f"[web_demo] No vec_normalize.pkl found at {vec_pkl}, running without normalization.")

    print("[web_demo] Starting evaluation loop ...")
    with STATE_LOCK:
        LIVE_STATE["status"] = "running"

    episode_count = 0
    infinite = (num_episodes <= 0)

    while infinite or episode_count < num_episodes:
        obs = vec_env.reset()
        ep_reward = 0.0
        ep_step = 0
        ep_reach = False
        ep_grasp = False
        ep_place = False

        episode_count += 1

        with STATE_LOCK:
            LIVE_STATE["episode"] = episode_count
            LIVE_STATE["step"] = 0
            LIVE_STATE["cumulative_reward"] = 0.0
            LIVE_STATE["reach_success"] = False
            LIVE_STATE["grasp_success"] = False
            LIVE_STATE["place_success"] = False

        done = False
        while not done:
            # obs shape: (1, 20) from DummyVecEnv
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = vec_env.step(action)

            reward = float(rewards[0])
            info = infos[0]
            ep_reward += reward
            ep_step += 1

            # Parse 20D obs for display: indices as per env docstring
            # [ee_pos(3), obj_pos(3), relative_pos(3), joint_pos(5), joint_vel(5), gripper(1)]
            raw_obs = obs[0]  # shape (20,)
            ee_pos   = raw_obs[0:3].tolist()
            obj_pos  = raw_obs[3:6].tolist()
            gripper  = float(raw_obs[19])

            ep_reach = ep_reach or bool(info.get("reach_success", False))
            ep_grasp = ep_grasp or bool(info.get("grasp_success", False))
            ep_place = ep_place or bool(info.get("place_success", False))

            with STATE_LOCK:
                LIVE_STATE["step"] = ep_step
                LIVE_STATE["reward"] = reward
                LIVE_STATE["cumulative_reward"] = ep_reward
                LIVE_STATE["ee_pos"] = ee_pos
                LIVE_STATE["obj_pos"] = obj_pos
                LIVE_STATE["gripper"] = gripper
                LIVE_STATE["reach_success"] = ep_reach
                LIVE_STATE["grasp_success"] = ep_grasp
                LIVE_STATE["place_success"] = ep_place

            done = bool(dones[0])

        # Episode done — append to history
        ep_record = {
            "episode":       episode_count,
            "steps":         ep_step,
            "reward":        ep_reward,
            "reach_success": ep_reach,
            "grasp_success": ep_grasp,
            "place_success": ep_place,
        }
        with STATE_LOCK:
            EPISODE_HISTORY.append(ep_record)

        print(
            f"[web_demo] Ep {episode_count:4d} | "
            f"steps={ep_step:3d} | reward={ep_reward:7.2f} | "
            f"reach={ep_reach} grasp={ep_grasp} place={ep_place}"
        )

    vec_env.close()
    print("[web_demo] Evaluation loop finished.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="6-DOF Arm RL live evaluation dashboard (stdlib HTTP)"
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Path to model .zip (without extension). "
             "Defaults to models/phase_{phase}/best_model.",
    )
    parser.add_argument(
        "--phase",
        type=int,
        default=2,
        help="Curriculum phase (0/1/2) used to pick model dir. Default: 2",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to serve the dashboard on. Default: 8080",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=0,
        help="Number of episodes to run (0 = infinite). Default: 0",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  6-DOF Arm RL Web Dashboard")
    print(f"  Phase  : {args.phase}")
    print(f"  Model  : {args.model or f'models/phase_{args.phase}/best_model (auto)'}")
    print(f"  Port   : {args.port}")
    print(f"  URL    : http://localhost:{args.port}")
    print("=" * 60)

    # Start evaluation loop in a daemon thread
    eval_thread = threading.Thread(
        target=eval_loop,
        args=(args.phase, args.model, args.episodes),
        daemon=True,
        name="eval-loop",
    )
    eval_thread.start()

    # Start web server in main thread (blocks until Ctrl-C)
    server = HTTPServer(("0.0.0.0", args.port), DashboardHandler)
    print(f"[web_demo] Serving dashboard at http://localhost:{args.port}  (Ctrl-C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[web_demo] Shutting down.")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
