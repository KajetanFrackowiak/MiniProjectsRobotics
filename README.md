# MiniProjectsRobotics

A collection of mini robotic projects.

## Projects
- **[Door opening](./door_opening)**: Multi-step task involving opening two cupboard doors, lifting an object, and placing it on one of the cupboard shelves using a KUKA LBR iiwa.
- **[Peg in a hole](./peg_in_a_hole)**: Simple peg-in-hole task performed with a KUKA LBR iiwa.
- **[Clutter clearing](./clutter_clearing)**: Picking and placing an arbitrary number of objects from one box into another using point clouds and a state machine, with a KUKA LBR iiwa.
- **[Bledt Kim](./bledt_kim/)**: Reimplementation of the methods described by [Bledt et al. (2018)](https://ieeexplore.ieee.org/document/8460904) and [Kim et al. (2019)](https://arxiv.org/abs/1909.06586) for the Mini Cheetah.

## Setup
This repository uses [uv](https://astral.sh/uv/) for dependency management.

```bash
# Install dependencies for a specific project
cd <project> && uv sync
```

## License
Licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.
