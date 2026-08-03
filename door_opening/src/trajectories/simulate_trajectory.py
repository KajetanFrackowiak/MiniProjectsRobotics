import numpy as np
from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import MakeHardwareStation
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, TrajectorySource
from pydrake.trajectories import Trajectory

from scenarios.scenario import load_iiwa_scenario
from utils.visualize import make_diagram


def build_and_simulate_trajectory(
    meshcat,
    q_traj: Trajectory,
    g_traj: Trajectory,
    duration: float = 0.01,
    diagram_name: str = "iiwa_diagram",
) -> tuple[Simulator, MultibodyPlant]:
    """Simulate trajectory for manipulation station.
    @param q_traj: Trajectory class used to initialize TrajectorySource for joints.
    @param g_traj: Trajectory class used to initialize TrajectorySource for gripper.
    """
    builder = DiagramBuilder()
    scenario = load_iiwa_scenario()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    q_traj_system = builder.AddSystem(TrajectorySource(q_traj))
    g_traj_system = builder.AddSystem(TrajectorySource(g_traj))

    wsg_force = builder.AddSystem(
        ConstantVectorSource(np.array([80.0], dtype=np.float64))
    )

    builder.Connect(
        q_traj_system.get_output_port(), station.GetInputPort("iiwa.position")
    )
    builder.Connect(
        g_traj_system.get_output_port(), station.GetInputPort("wsg.position")
    )
    builder.Connect(
        wsg_force.get_output_port(), station.GetInputPort("wsg.force_limit")
    )

    diagram = builder.Build()
    make_diagram(diagram, name=diagram_name)

    simulator = Simulator(diagram)
    meshcat.StartRecording(set_visualizations_while_recording=False)
    simulator.AdvanceTo(duration)
    print(f"Simulation finished ({duration} s), publishing recording...")
    meshcat.PublishRecording()

    return simulator, plant
