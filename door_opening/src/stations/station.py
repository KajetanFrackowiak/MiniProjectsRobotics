import time

import numpy as np
from manipulation.scenarios import AddMultibodyTriad
from manipulation.station import MakeHardwareStation
from pydrake.math import RigidTransform
from pydrake.multibody.inverse_kinematics import (
    DifferentialInverseKinematicsIntegrator,
    DifferentialInverseKinematicsParameters,
)
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import ConstantVectorSource, MatrixGain
from pydrake.visualization import MeshcatPoseSliders

from controllers.iiwa_controller import CreateIiwaPlant
from scenarios.scenario import load_iiwa_scenario
from utils.visualize import make_diagram


def setup_manipulation_station(meshcat) -> RigidTransform:
    builder = DiagramBuilder()
    scenario = load_iiwa_scenario()
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    iiwa_position = builder.AddSystem(
        ConstantVectorSource(np.array([0.0, 0.6, 0.0, -1.75, 0.0, 1.0, 0.0]))
    )
    builder.Connect(
        iiwa_position.get_output_port(), station.GetInputPort("iiwa.position")
    )

    wsg_position = builder.AddSystem(ConstantVectorSource([0.06]))
    builder.Connect(
        wsg_position.get_output_port(), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()
    make_diagram(diagram, name="test_station")

    context = plant.CreateDefaultContext()
    gripper = plant.GetBodyByName("body")

    initial_pose = plant.EvalBodyPoseInWorld(context, gripper)

    simulator = Simulator(diagram)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(5.0)

    return initial_pose


def teleop_inverse_kinematics(meshcat):
    builder = DiagramBuilder()
    scenario = load_iiwa_scenario(iiwa_collision=True)
    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    plant = station.GetSubsystemByName("plant")
    scene_graph = station.GetSubsystemByName("scene_graph")
    AddMultibodyTriad(plant.GetFrameByName("body"), scene_graph)

    plant_ctl, _ = CreateIiwaPlant()
    params = DifferentialInverseKinematicsParameters(
        plant_ctl.num_positions(), plant_ctl.num_velocities()
    )
    q0_ctl = plant_ctl.GetPositions(plant_ctl.CreateDefaultContext())
    params.set_nominal_joint_position(q0_ctl)
    params.set_end_effector_angular_speed_limit(2)
    params.set_end_effector_translational_velocity_limits([-2, -2, -2], [2, 2, 2])
    v_limits = np.array([1.4, 1.4, 1.7, 1.3, 2.2, 2.3, 2.3, 5.0, 5.0])
    params.set_joint_velocity_limits((-v_limits, v_limits))
    params.set_joint_centering_gain(10 * np.eye(9))

    differential_ik = builder.AddSystem(
        DifferentialInverseKinematicsIntegrator(
            plant_ctl,
            plant_ctl.GetFrameByName("body"),
            0.005,
            params,
            log_only_when_result_state_changes=True,
        )
    )
    differential_ik.set_name("interactive_ik")

    meshcat.DeleteAddedControls()
    sliders = builder.AddSystem(MeshcatPoseSliders(meshcat))
    builder.Connect(
        sliders.get_output_port(0), differential_ik.GetInputPort("X_AE_desired")
    )

    C_iiwa = np.zeros((7, 9))
    C_iiwa[:7, :7] = np.eye(7)
    q_selector = builder.AddSystem(MatrixGain(C_iiwa))
    builder.Connect(
        differential_ik.GetOutputPort("joint_positions"),
        q_selector.get_input_port(),
    )
    builder.Connect(q_selector.get_output_port(), station.GetInputPort("iiwa.position"))

    wsg_position = builder.AddSystem(ConstantVectorSource([0.0]))
    builder.Connect(
        wsg_position.get_output_port(), station.GetInputPort("wsg.position")
    )

    diagram = builder.Build()
    make_diagram(diagram, name="teleop_ik")
    root_context = diagram.CreateDefaultContext()
    differential_ik.SetPositions(
        differential_ik.GetMyMutableContextFromRoot(root_context), q0_ctl
    )
    plant_context = plant.GetMyContextFromRoot(root_context)
    simulator = Simulator(diagram, root_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    X_start = plant.CalcRelativeTransform(
        plant_context, plant.world_frame(), plant.GetFrameByName("body")
    )
    sliders.SetPose(X_start)

    meshcat.AddButton("Close Gripper", "KeyC")
    meshcat.AddButton("Open Gripper", "KeyV")
    meshcat.AddButton("Stop Teleop", "Escape")
    print(
        "Move the pose sliders in Meshcat. Use 'Close Gripper' / 'Open Gripper' "
        "to control the wsg. Press 'Stop Teleop' to exit."
    )

    q_wsg = 0.0
    finger_step = 0.002
    last_close_clicks = 0
    last_open_clicks = 0
    wsg_source_context = wsg_position.GetMyMutableContextFromRoot(root_context)
    dt = 0.01
    t = 0.0

    try:
        while meshcat.GetButtonClicks("Stop Teleop") < 1:
            close_clicks = meshcat.GetButtonClicks("Close Gripper")
            open_clicks = meshcat.GetButtonClicks("Open Gripper")
            if close_clicks > last_close_clicks:
                last_close_clicks = close_clicks
                q_wsg = max(0.0, q_wsg - finger_step)
                print(f"finger position: {q_wsg:.4f}")
            if open_clicks > last_open_clicks:
                last_open_clicks = open_clicks
                q_wsg = min(0.055, q_wsg + finger_step)
                print(f"finger position: {q_wsg:.4f}")
            wsg_position.get_mutable_source_value(wsg_source_context).set_value([q_wsg])
            t += dt
            simulator.AdvanceTo(t)
            time.sleep(dt)
    finally:
        meshcat.DeleteButton("Close Gripper")
        meshcat.DeleteButton("Open Gripper")
        meshcat.DeleteButton("Stop Teleop")
