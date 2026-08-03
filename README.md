# MiniProjectsRobotics

A collection of mini robotic projects.

## Projects
- **[Door opening](./door_opening)**: Multi-step task — opening two cupboard doors, lifting an object, and placing it on one of the cupboard's shelves with a KUKA LBR iiwa
- **[Peg in a hole](./peg_in_a_hole)**: Simple peg-in-a-hole task with a KUKA LBR iiwa
- **[Clutter clearing](./clutter_clearing)**: Picking and placing any number of objects from one box into another, using point clouds and a state machine, with a KUKA LBR iiwa

## Setup
This repository uses [uv](https://astral.sh/uv/) for dependency management.

```bash
# Install dependencies for a specific project
cd <project> && uv sync
```

## License
Licensed under the **MIT License**. See the [LICENSE](LICENSE) file for the full text.
