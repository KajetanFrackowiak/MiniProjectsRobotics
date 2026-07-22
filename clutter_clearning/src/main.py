import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT / "src"))

from pydrake.geometry import StartMeshcat
from simulations.clutter_demo import clutter_clearing_demo

def main():
    parser = argparse.ArgumentParser(description="Clutter Clearing Simulation")
    parser.add_argument("--num-objects", type=int, default=6, help="Number of YCB objects to spawn")
    parser.add_argument("--seed", type=int, default=135, help="Random seed")

    args = parser.parse_args()

    # Start Meshcat visualizer
    meshcat = StartMeshcat()
    print(f"Meshcat URL: {meshcat.web_url()}")

    # Run simulation demo
    clutter_clearing_demo(
        meshcat=meshcat,
        num_objects=args.num_objects,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()