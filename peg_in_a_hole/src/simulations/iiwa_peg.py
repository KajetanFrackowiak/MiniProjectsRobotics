import os
import numpy as np
from typing import Callable, Literal

from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.analysis import Simulator
from pydrake.systems.primitives import TrajectorySource

from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import LoadScenario, MakeHardwareStation

from scenerios.iiwa import make_scenario_string
from utils.trajectory import get_peg_frame_offset, get_peg_insertion_trajectory
from utils.visualization import add_setpoint_visualization
from controllers.operational_stifness import OperationalStiffnessController
from solvers.ik import solve_ik

def run_iiwa_peg_simulation(
    k_p: np.ndarray,
    k_d: np.ndarray,
    meshcat: Callable,
    peg_frame_placement: Literal["back", "center", "tip"] = "center",
) -> None:

    # Set up the scene
    peg_frame_offset = get_peg_frame_offset(peg_frame_placement)
    scenario_string = make_scenario_string(peg_frame_placement)
    scenario = LoadScenario(data=scenario_string)
    builder = DiagramBuilder()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("peg_tip_frame"), scene_graph)

    # Compute the peg insertion trajectory
    pos_xz_start = np.array([0.75, 0.4])
    start_pose, traj = get_peg_insertion_trajectory(
        pos_xz_start, peg_frame_offset, trans_pert=0.07, rot_pert=0.02
    )
    pos_ref = builder.AddSystem(TrajectorySource(traj))
    vel_ref = builder.AddSystem(TrajectorySource(traj.derivative()))
    add_setpoint_visualization(
        builder, meshcat, pos_ref.get_output_port(), peg_frame_offset
    )


    controller = builder.AddSystem(OperationalStiffnessController(plant, k_p, k_d))
    builder.Connect(pos_ref.get_output_port(), controller.get_input_port(0)) # [x, z, theta]
    builder.Connect(vel_ref.get_output_port(), controller.get_input_port(1)) # [x_dot, z_dot, theta_dot]
    
    builder.Connect(station.GetOutputPort("iiwa.state_estimated"), controller.get_input_port(2))
    builder.Connect(controller.get_output_port(0), station.GetInputPort("iiwa.torque"))

    q_initial = solve_ik(plant, start_pose, trans_tol=1e-3, ang_tol=1e-3)

    diagram = builder.Build()

    os.makedirs("diagrams", exist_ok=True)
    with open("diagrams/iiwa_peg.dot", "w") as f:
        f.write(diagram.GetGraphvizString())

    simulator = Simulator(diagram)
    context = simulator.get_mutable_context()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    iiwa = plant.GetModelInstanceByName("iiwa")
    plant.SetPositions(plant_context, iiwa, q_initial)
    plant.SetVelocities(plant_context, iiwa, np.zeros(plant.num_velocities(iiwa))) # start at rest
    

    simulator.set_target_realtime_rate(1.0)
    meshcat.StartRecording()
    simulator.AdvanceTo(traj.end_time())
    meshcat.PublishRecording()
    
