import os
from collections.abc import Callable
from typing import Literal

import numpy as np
from manipulation.scenarios import AddMultibodyTriad
from pydrake.geometry import MeshcatVisualizer
from pydrake.multibody.meshcat import ContactVisualizer, ContactVisualizerParams
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import TrajectorySource

from controllers.stiffness import StiffnessController
from utils.peg_geometry import (
    add_planar_hole_to_plant,
    add_planar_peg_to_plant,
    add_table_to_plant,
)
from utils.trajectory import get_peg_frame_offset, get_peg_insertion_trajectory
from utils.visualization import add_setpoint_visualization


def run_planar_peg_simulation(
    k_p: np.ndarray,
    k_d: np.ndarray,
    meshcat: Callable,
    peg_frame_placement: Literal["back", "center", "tip"] = "center",
) -> None:

    # Compute the numeric offset of the peg frame based on desired frame placement
    peg_frame_offset = get_peg_frame_offset(peg_frame_placement)

    # Set up the scene for the planar floating peg.
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.005)
    _peg = add_planar_peg_to_plant(plant, peg_frame_offset=peg_frame_offset)
    table = add_table_to_plant(plant)
    _hole = add_planar_hole_to_plant(plant, table)
    plant.Finalize()

    # Visualize the peg tip
    AddMultibodyTriad(plant.GetFrameByName("peg_tip_frame"), scene_graph)

    # Set up meshcat
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)
    ContactVisualizer.AddToBuilder(
        builder,
        plant,
        meshcat,
        ContactVisualizerParams(radius=0.005, newtons_per_meter=10.0),
    )

    # Set up trajectory references for the peg
    start_pos, traj = get_peg_insertion_trajectory(
        np.array([0.5, 0.35]), peg_frame_offset
    )
    pos_ref = builder.AddSystem(TrajectorySource(traj))
    vel_ref = builder.AddSystem(TrajectorySource(traj.derivative()))
    add_setpoint_visualization(
        builder, meshcat, pos_ref.get_output_port(), peg_frame_offset=peg_frame_offset
    )

    controller = builder.AddSystem(StiffnessController(plant, k_p, k_d))
    builder.Connect(pos_ref.get_output_port(), controller._q_des_port)
    builder.Connect(vel_ref.get_output_port(), controller._q_dot_des_port)
    builder.Connect(plant.get_state_output_port(), controller._state_port)
    builder.Connect(controller.get_output_port(), plant.get_actuation_input_port())

    # Build diagram and simulate
    diagram = builder.Build()

    os.makedirs("diagrams", exist_ok=True)
    with open("diagrams/planar_peg.dot", "w") as f:
        f.write(diagram.GetGraphvizString())

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyContextFromRoot(context)

    plant.SetPositions(plant_context, start_pos)

    simulator.set_target_realtime_rate(1.0)
    meshcat.StartRecording()
    simulator.AdvanceTo(7.5)  # feel free to change the end time
    meshcat.StopRecording()
    meshcat.PublishRecording()
