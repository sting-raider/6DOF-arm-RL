import argparse
import os
import sys

# Insert project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.app import AppLauncher

argparser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(argparser)
args_cli = argparser.parse_args([]) # pass empty list to parse_args
app_launcher = AppLauncher(args_cli)
app = app_launcher.app

from isaaclab.envs.mdp import RelativeJointPositionActionCfg, BinaryJointPositionActionCfg
print("Successfully imported RelativeJointPositionActionCfg and BinaryJointPositionActionCfg!")
print(RelativeJointPositionActionCfg, BinaryJointPositionActionCfg)

app.close()
