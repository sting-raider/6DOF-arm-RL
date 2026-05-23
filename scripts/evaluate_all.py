#!/usr/bin/env python3
"""
Batch evaluation script — runs all 3 curriculum phases sequentially
and prints a summary comparison table.

Usage:
    python scripts/evaluate_all.py              # Real run (evaluates all phases)
    python scripts/evaluate_all.py --dry-run    # Print planned commands only

Environment variables:
    HERMES_DISABLE_DR=1  Disables domain randomization (set for Phases 0 & 1)
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

PHASES = [
    {
        "id": 0,
        "name": "REACH",
        "model": "models/phase_0/best_model",
        "episodes": 10,
        "disable_dr": True,
        "description": "Phase 0 — Reach target (no DR)",
    },
    {
        "id": 1,
        "name": "GRASP",
        "model": "models/phase_1/best_model",
        "episodes": 10,
        "disable_dr": True,
        "description": "Phase 1 — Reach + Grasp (no DR)",
    },
    {
        "id": 2,
        "name": "PLACE",
        "model": "models/phase_2/best_model",
        "episodes": 20,
        "disable_dr": False,
        "description": "Phase 2 — Full pick-and-place (DR enabled)",
    },
]


def build_command(phase_cfg):
    """Build a shell command list for the given phase config."""
    cmd = []
    if phase_cfg["disable_dr"]:
        cmd = ["env", "HERMES_DISABLE_DR=1"]

    cmd += [
        sys.executable or "python",
        "scripts/evaluate_comprehensive.py",
        "--model", phase_cfg["model"],
        "--phase", str(phase_cfg["id"]),
        "--episodes", str(phase_cfg["episodes"]),
    ]
    return cmd


def run_phase(phase_cfg):
    """Execute evaluation for one phase and return stdout+stderr text."""
    cmd = build_command(phase_cfg)
    cmd_str = " ".join(str(c) if " " not in str(c) else f"'{c}'" for c in cmd)

    print(f"\n{'=' * 60}")
    print(f"▶  {phase_cfg['description']}")
    print(f"   Command: {cmd_str}")
    print(f"{'=' * 60}\n")

    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=600,  # 10 min per phase max
    )

    # Print stdout
    print(result.stdout)

    # Print stderr if non-empty
    if result.stderr.strip():
        print("--- stderr ---")
        print(result.stderr.strip())

    if result.returncode != 0:
        print(f"⚠️  Phase {phase_cfg['id']} exited with code {result.returncode}")

    return result.stdout + "\n" + result.stderr


def parse_results(output: str) -> dict:
    """Extract evaluation metrics from the output text."""
    metrics = {}
    patterns = {
        "mean_reward": r"Mean reward:\s+([\d.-]+)\s*±\s*([\d.]+)",
        "episode_length": r"Episode length:\s+([\d.]+)\s*±\s*([\d.]+)",
        "reach_success": r"Reach\s+success:\s+(\d+)/(\d+)\s+\((\d+)%\)",
        "grasp_success": r"Grasp\s+success:\s+(\d+)/(\d+)\s+\((\d+)%\)",
        "place_success": r"Place\s+success:\s+(\d+)/(\d+)\s+\((\d+)%\)",
    }
    for key, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            metrics[key] = {
                "value": m.group(1),
                "std": m.group(2) if key in ("mean_reward", "episode_length") else None,
                "numerator": m.group(1) if key not in ("mean_reward", "episode_length") else None,
                "denominator": m.group(2) if key not in ("mean_reward", "episode_length") else None,
                "pct": m.group(3) if key not in ("mean_reward", "episode_length") else None,
            }
    return metrics


def print_summary(all_results):
    """Print a formatted comparison table of all 3 phases."""
    print(f"\n{'=' * 64}")
    print("  BATCH EVALUATION SUMMARY")
    print(f"{'=' * 64}")

    # Header
    header = f"{'Metric':<22} {'Phase 0 (REACH)':<14} {'Phase 1 (GRASP)':<14} {'Phase 2 (PLACE)':<14}"
    print(header)
    print("-" * 64)

    # Reward row
    reward_strs = []
    for phase_id in range(3):
        m = all_results.get(phase_id, {}).get("mean_reward")
        if m:
            reward_strs.append(f"{m['value']} ± {m['std']}")
        else:
            reward_strs.append("N/A")
    print(f"{'Mean Reward':<22} {reward_strs[0]:<14} {reward_strs[1]:<14} {reward_strs[2]:<14}")

    # Episode length row
    len_strs = []
    for phase_id in range(3):
        m = all_results.get(phase_id, {}).get("episode_length")
        if m:
            len_strs.append(f"{m['value']} ± {m['std']}")
        else:
            len_strs.append("N/A")
    print(f"{'Episode Length':<22} {len_strs[0]:<14} {len_strs[1]:<14} {len_strs[2]:<14}")

    # Success rates
    for label, key in [("Reach Success", "reach_success"),
                       ("Grasp Success", "grasp_success"),
                       ("Place Success", "place_success")]:
        strs = []
        for phase_id in range(3):
            m = all_results.get(phase_id, {}).get(key)
            if m:
                strs.append(f"{m['numerator']}/{m['denominator']} ({m['pct']}%)")
            else:
                strs.append("N/A")
        print(f"{label:<22} {strs[0]:<14} {strs[1]:<14} {strs[2]:<14}")

    print("-" * 64)
    print()


def dry_run():
    """Print what the script would do without actually running anything."""
    print(f"\n{'=' * 60}")
    print("  DRY RUN — No commands will be executed")
    print(f"{'=' * 60}\n")

    for cfg in PHASES:
        cmd = build_command(cfg)
        cmd_str = " ".join(str(c) if " " not in str(c) else f"'{c}'" for c in cmd)
        dr_status = "DR DISABLED" if cfg["disable_dr"] else "DR ENABLED"
        print(f"  [{cfg['name']}] {cfg['description']}")
        print(f"  Model:    {cfg['model']}")
        print(f"  Episodes: {cfg['episodes']}")
        print(f"  DR:       {dr_status}")
        print(f"  Command:  {cmd_str}")
        print()

    # Check if model files exist
    print(f"  --- Model file checks ---")
    for cfg in PHASES:
        model_path = PROJECT_ROOT / cfg["model"]
        model_zip = model_path.with_suffix(".zip")
        if model_zip.exists():
            print(f"  ✅ {cfg['model']}.zip — FOUND ({model_zip.stat().st_size / 1e6:.1f} MB)")
        elif model_path.exists():
            print(f"  ✅ {cfg['model']} — FOUND")
        else:
            print(f"  ⚠️  {cfg['model']}.zip — MISSING (training may still be running)")

    print(f"\n  {'=' * 60}")
    print(f"  DRY RUN COMPLETE — No commands were executed")
    print(f"  {'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluation of all 3 curriculum phases."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without executing them",
    )
    args = parser.parse_args()

    if args.dry_run:
        dry_run()
        return

    os.chdir(str(PROJECT_ROOT))

    all_results = {}

    for cfg in PHASES:
        output = run_phase(cfg)
        metrics = parse_results(output)
        all_results[cfg["id"]] = metrics

        # Re-save results (evaluate_comprehensive already does this,
        # but we ensure we have the canonical version)
        results_path = PROJECT_ROOT / "videos" / f"phase_{cfg['id']}_evaluation_results.txt"
        if results_path.exists():
            print(f"  → Results saved to {results_path}")
        else:
            print(f"  ⚠️  Expected results file not found: {results_path}")

    # Print cross-phase summary table
    print_summary(all_results)

    print("✅ Batch evaluation complete.")


if __name__ == "__main__":
    main()
