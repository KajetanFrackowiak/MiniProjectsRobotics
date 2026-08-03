import argparse

import numpy as np
from pydrake.geometry import StartMeshcat

from simulations.iiwa_peg import run_iiwa_peg_simulation
from simulations.planar_peg import run_planar_peg_simulation


def main():
    parser = argparse.ArgumentParser(description="Run peg-in-hole simulations.")

    parser.add_argument(
        "--sim",
        choices=["planar", "iiwa"],
        default="iiwa",
    )

    args = parser.parse_args()

    meshcat = StartMeshcat()

    k_p = np.array([100, 500, 10])
    k_d = np.array([20, 45, 2])

    if args.sim == "planar":
        k_p = k_p * 0.1
        k_d = k_d * 0.1
        run_planar_peg_simulation(k_p, k_d, meshcat, peg_frame_placement="center")
    elif args.sim == "iiwa":
        run_iiwa_peg_simulation(k_p, k_d, meshcat, peg_frame_placement="center")


if __name__ == "__main__":
    main()
