#!/usr/bin/env python3
"""
Web-based interactive demo for the 6-DOF arm.
Run this and open http://localhost:5000 in your browser.

Controls:
- Click to place the object
- Drag to move it
- Buttons: Grasp, Place, Reset, Auto
"""

import sys
import os
import numpy as np
import threading
import base64
from io import BytesIO
from pathlib import Path
from flask import Flask, render_template_string, Response, jsonify, request

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import SAC
from envs.pick_and_place_env import PickAndPlaceEnv
from robots.kuka_iiwa import KukaRobot
from utils.constants import TABLE_HEIGHT, BASKET_POS

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XML_PATH = os.path.join(PROJECT_DIR, "scenes", "pick_and_place_scene.xml")
MODEL_PATH = os.path.join(PROJECT_DIR, "models", "phase_2", "final_model")

app = Flask(__name__)

# Global state
robot = None
model = None
lock = threading.Lock()


def init_robot():
    """Initialize the robot and model."""
    global robot, model
    robot = KukaRobot(XML_PATH)
    if os.path.exists(MODEL_PATH):
        model = SAC.load(MODEL_PATH)
    robot.set_object_position(np.array([0.2, 0.0, TABLE_HEIGHT + 0.021]))


@app.route('/')
def index():
    return render_template_string("""
    <!DOCTYPE html>
    <html>
    <head>
        <title>6-DOF Arm Interactive Demo</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body {
                font-family: 'Segoe UI', system-ui, sans-serif;
                background: #1a1a2e; color: #eee;
                display: flex; flex-direction: column; align-items: center;
                min-height: 100vh; padding: 20px;
            }
            h1 { color: #00ff88; margin-bottom: 10px; font-size: 24px; }
            .container {
                display: flex; gap: 20px; flex-wrap: wrap;
                justify-content: center; width: 100%; max-width: 1100px;
            }
            .scene-panel {
                background: #16213e; border-radius: 12px; padding: 15px;
                border: 1px solid #0f3460; flex: 1; min-width: 640px;
            }
            .scene-panel h2 { color: #00ff88; margin-bottom: 10px; font-size: 18px; }
            #scene-video {
                width: 100%; border-radius: 8px;
                border: 2px solid #0f3460;
                background: #000; image-rendering: pixelated;
            }
            .controls-panel {
                background: #16213e; border-radius: 12px; padding: 15px;
                border: 1px solid #0f3460; min-width: 300px; flex: 0;
            }
            .controls-panel h2 { color: #00ff88; margin-bottom: 10px; font-size: 18px; }
            .btn-grid {
                display: grid; grid-template-columns: 1fr 1fr;
                gap: 8px; margin-bottom: 15px;
            }
            .btn {
                padding: 12px 16px; border: none; border-radius: 8px;
                font-size: 14px; font-weight: 600; cursor: pointer;
                transition: all 0.2s; color: white;
            }
            .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.3); }
            .btn:active { transform: translateY(0); }
            .btn-grasp { background: #e94560; }
            .btn-grasp:hover { background: #ff6b81; }
            .btn-place { background: #0f3460; }
            .btn-place:hover { background: #1a5276; }
            .btn-reset { background: #533483; }
            .btn-reset:hover { background: #6c4fa8; }
            .btn-auto { background: #00b894; }
            .btn-auto:hover { background: #00d2a0; }
            .btn-random { background: #fdcb6e; color: #2d3436; }
            .btn-random:hover { background: #ffeaa7; }
            .status-box {
                background: #0a0a23; border-radius: 8px; padding: 12px;
                margin-bottom: 15px; font-family: 'Courier New', monospace;
                font-size: 13px; line-height: 1.6;
            }
            .status-box div { margin: 2px 0; }
            .label { color: #636e72; }
            .value { color: #00ff88; }
            .value.grasped { color: #e94560; }
            .pos-input { 
                display: flex; gap: 8px; margin: 10px 0;
                align-items: center;
            }
            .pos-input input {
                background: #0a0a23; border: 1px solid #0f3460;
                color: #eee; padding: 6px 8px; border-radius: 4px;
                width: 70px; font-size: 13px;
            }
            .pos-input button {
                background: #0f3460; color: white; border: none;
                padding: 6px 12px; border-radius: 4px; cursor: pointer;
            }
            .pos-input button:hover { background: #1a5276; }
            .obj-label { font-size: 12px; color: #636e72; }
        </style>
    </head>
    <body>
        <h1>🦾 6-DOF Arm Interactive Demo</h1>
        <div class="container">
            <div class="scene-panel">
                <h2>📷 Live Scene View</h2>
                <img id="scene-video" src="/video_feed" alt="Live scene">
            </div>
            <div class="controls-panel">
                <h2>🎮 Controls</h2>
                <div class="btn-grid">
                    <button class="btn btn-grasp" onclick="sendAction('grasp')">🤏 Grasp</button>
                    <button class="btn btn-place" onclick="sendAction('place')">📦 Place</button>
                    <button class="btn btn-reset" onclick="sendAction('reset')">🔄 Reset</button>
                    <button class="btn btn-auto" onclick="sendAction('auto')">🤖 Auto</button>
                    <button class="btn btn-random" onclick="sendAction('random')">🎲 Random</button>
                    <button class="btn btn-grasp" onclick="sendAction('release')">✋ Release</button>
                </div>
                
                <h3 style="color:#00ff88; font-size:14px; margin-top:15px;">📍 Object Position</h3>
                <div class="pos-input">
                    <span class="obj-label">X:</span>
                    <input type="number" id="obj-x" step="0.01" value="0.2">
                    <span class="obj-label">Y:</span>
                    <input type="number" id="obj-y" step="0.01" value="0.0">
                    <button onclick="setPosition()">Set</button>
                </div>
                <p class="obj-label">Object position on table<br>X: 0.05-0.35, Y: -0.15-0.15</p>
                
                <div class="status-box" id="status">
                    <div><span class="label">Object:</span> <span class="value" id="obj-pos">--</span></div>
                    <div><span class="label">EE:</span> <span class="value" id="ee-pos">--</span></div>
                    <div><span class="label">Grasped:</span> <span class="value" id="grasped">--</span></div>
                    <div><span class="label">Distance:</span> <span class="value" id="distance">--</span></div>
                    <div><span class="label">Steps:</span> <span class="value" id="steps">0</span></div>
                </div>
            </div>
        </div>

        <script>
            const video = document.getElementById('scene-video');
            const statusDiv = document.getElementById('status');

            function sendAction(action) {
                fetch('/action/' + action).then(r => r.json()).then(updateStatus);
            }

            function setPosition() {
                const x = document.getElementById('obj-x').value;
                const y = document.getElementById('obj-y').value;
                fetch('/set_position/' + x + '/' + y).then(r => r.json()).then(updateStatus);
            }

            function updateStatus(data) {
                if (!data) return;
                document.getElementById('obj-pos').textContent = 
                    `(${data.object_pos[0].toFixed(3)}, ${data.object_pos[1].toFixed(3)}, ${data.object_pos[2].toFixed(3)})`;
                document.getElementById('ee-pos').textContent = 
                    `(${data.ee_pos[0].toFixed(3)}, ${data.ee_pos[1].toFixed(3)}, ${data.ee_pos[2].toFixed(3)})`;
                const graspedEl = document.getElementById('grasped');
                graspedEl.textContent = data.grasped ? 'YES' : 'no';
                graspedEl.className = 'value' + (data.grasped ? ' grasped' : '');
                document.getElementById('distance').textContent = data.distance.toFixed(4);
                document.getElementById('steps').textContent = data.steps;
            }

            // Update status every second
            setInterval(() => {
                fetch('/status').then(r => r.json()).then(updateStatus);
            }, 1000);
        </script>
    </body>
    </html>
    """)


def render_frame():
    """Render a frame and return as JPEG bytes."""
    global robot
    with lock:
        if robot is None:
            return None
        img = robot.render_image(width=640, height=480)
        # Convert RGB to BGR for JPEG (OpenCV uses BGR)
        import cv2
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # Add overlay text
        obj_pos = robot.get_object_pos()
        ee_pos = robot.get_ee_pos()
        grasped = robot.is_object_grasped()
        
        cv2.putText(img_bgr, f"Obj: ({obj_pos[0]:.2f}, {obj_pos[1]:.2f}, {obj_pos[2]:.2f})",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_bgr, f"EE: ({ee_pos[0]:.2f}, {ee_pos[1]:.2f}, {ee_pos[2]:.2f})",
                   (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(img_bgr, f"Grasped: {grasped}",
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 0, 255) if grasped else (0, 255, 0), 1)
        
        # Draw basket target
        basket_center = (int(640 * 0.5), int(480 * 0.5))  # Approximate
        cv2.drawMarker(img_bgr, (320, 240), (255, 165, 0), cv2.MARKER_DIAMOND, 20, 2)
        cv2.putText(img_bgr, "BASKET", (310, 230), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 165, 0), 1)
        
        ret, jpeg = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpeg.tobytes()


@app.route('/video_feed')
def video_feed():
    """Stream MJPEG video."""
    def generate():
        while True:
            frame = render_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            import time
            time.sleep(0.05)  # ~20 FPS
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def get_status():
    """Return current state as JSON."""
    global robot
    with lock:
        if robot is None:
            return jsonify({'error': 'not initialized'})
        obj_pos = robot.get_object_pos().tolist()
        ee_pos = robot.get_ee_pos().tolist()
        grasped = robot.is_object_grasped()
        dist = float(np.linalg.norm(np.array(obj_pos) - np.array(ee_pos)))
        return jsonify({
            'object_pos': obj_pos,
            'ee_pos': ee_pos,
            'grasped': grasped,
            'distance': dist,
            'steps': 0
        })


@app.route('/action/<action>')
def do_action(action):
    """Handle user actions."""
    global robot, model
    with lock:
        if robot is None:
            return jsonify({'error': 'not initialized'})
        
        if action == 'grasp':
            # Close gripper - try to grasp
            robot.apply_action(np.array([0, 0, 0, 0, 0, 1.0]))
            
        elif action == 'release':
            # Open gripper
            robot.apply_action(np.array([0, 0, 0, 0, 0, -1.0]))
            
        elif action == 'place':
            # Move arm toward basket and release
            # Use learned policy for several steps
            if model is not None:
                for _ in range(50):
                    obs = np.concatenate([
                        robot.get_ee_pos(),
                        np.array([0.4, 0.0, TABLE_HEIGHT + 0.05]),  # Target: basket
                        np.array([robot.get_gripper_state()]),
                        np.array([1.0 if robot.is_object_grasped() else 0.0])
                    ]).astype(np.float32)
                    action, _ = model.predict(obs, deterministic=True)
                    robot.apply_action(action)
            else:
                # Manual place
                robot.apply_action(np.array([0.1, -0.1, 0.0, 0.1, 0.0, 1.0]))
            
        elif action == 'reset':
            robot.reset(object_pos=np.array([0.2, 0.0, TABLE_HEIGHT + 0.021]))
            
        elif action == 'auto':
            # Run a complete auto cycle
            for _ in range(100):
                if model is None:
                    break
                obs = np.concatenate([
                    robot.get_ee_pos(),
                    robot.get_object_pos(),
                    np.array([robot.get_gripper_state()]),
                    np.array([1.0 if robot.is_object_grasped() else 0.0])
                ]).astype(np.float32)
                action, _ = model.predict(obs, deterministic=True)
                robot.apply_action(action)
                if robot.is_object_in_basket():
                    break
                    
        elif action == 'random':
            # Random action
            import random
            action = np.random.uniform(-0.5, 0.5, size=6)
            robot.apply_action(action)
        
        # Return updated status
        obj_pos = robot.get_object_pos().tolist()
        ee_pos = robot.get_ee_pos().tolist()
        return jsonify({
            'object_pos': obj_pos,
            'ee_pos': ee_pos,
            'grasped': robot.is_object_grasped(),
            'distance': float(np.linalg.norm(np.array(obj_pos) - np.array(ee_pos))),
            'in_basket': robot.is_object_in_basket(),
            'steps': 0
        })


@app.route('/set_position/<x>/<y>')
def set_position(x, y):
    """Set object position on the table."""
    global robot
    with lock:
        if robot is None:
            return jsonify({'error': 'not initialized'})
        x = float(x)
        y = float(y)
        pos = np.array([x, y, TABLE_HEIGHT + 0.021])
        robot.set_object_position(pos)
        obj_pos = robot.get_object_pos().tolist()
        return jsonify({
            'object_pos': obj_pos,
            'success': True
        })


if __name__ == '__main__':
    init_robot()
    print("🚀 Interactive 6-DOF Arm Web Demo")
    print(f"    Open http://localhost:5000 in your browser")
    print(f"    Scene: {XML_PATH}")
    print(f"    Model: {MODEL_PATH}")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
