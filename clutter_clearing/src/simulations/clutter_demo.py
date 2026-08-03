import numpy as np
from manipulation import FindResource
from manipulation.meshcat_utils import StopButton
from manipulation.scenarios import AddIiwaDifferentialIK, ycb
from manipulation.station import (
    AddPointClouds,
    AppendDirectives,
    LoadScenario,
    MakeHardwareStation,
)
from pydrake.common import RandomGenerator
from pydrake.math import RigidTransform, UniformlyRandomRotationMatrix
from pydrake.systems.analysis import Simulator
from pydrake.systems.framework import DiagramBuilder
from pydrake.systems.primitives import PortSwitch

from perception.grasp_selector import GraspSelector
from planning.state_machine import Planner
from utils.assets import SCENARIO_PATH


def clutter_clearing_demo(
    meshcat,
    num_objects: int = 5,
    seed: int = 135,
):
    meshcat.Delete()
    builder = DiagramBuilder()

    rng = np.random.default_rng(seed)
    generator = RandomGenerator(rng.integers(0, 1000))

    scenario = LoadScenario(
        filename=FindResource(str(SCENARIO_PATH)),
        scenario_name="Clutter",
    )

    model_directives = "\ndirectives:\n"
    for i in range(num_objects):
        object_num = rng.integers(0, len(ycb))
        if "cracker_box" in ycb[object_num]:
            continue
        model_directives += f"""
- add_model:
    name: ycb{i}
    file: package://manipulation/hydro/{ycb[object_num]}
"""
    scenario = AppendDirectives(scenario, data=model_directives)

    station = builder.AddSystem(MakeHardwareStation(scenario, meshcat))
    to_point_cloud = AddPointClouds(scenario=scenario, station=station, builder=builder)
    plant = station.GetSubsystemByName("plant")

    y_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin0"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera0"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera1"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera2"))[0],
            ],
        )
    )
    builder.Connect(
        to_point_cloud["camera0"].get_output_port(),
        y_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud["camera1"].get_output_port(),
        y_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud["camera2"].get_output_port(),
        y_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        y_bin_grasp_selector.GetInputPort("body_poses"),
    )

    x_bin_grasp_selector = builder.AddSystem(
        GraspSelector(
            plant,
            plant.GetModelInstanceByName("bin1"),
            camera_body_indices=[
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera3"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera4"))[0],
                plant.GetBodyIndices(plant.GetModelInstanceByName("camera5"))[0],
            ],
        )
    )
    builder.Connect(
        to_point_cloud["camera3"].get_output_port(),
        x_bin_grasp_selector.get_input_port(0),
    )
    builder.Connect(
        to_point_cloud["camera4"].get_output_port(),
        x_bin_grasp_selector.get_input_port(1),
    )
    builder.Connect(
        to_point_cloud["camera5"].get_output_port(),
        x_bin_grasp_selector.get_input_port(2),
    )
    builder.Connect(
        station.GetOutputPort("body_poses"),
        x_bin_grasp_selector.GetInputPort("body_poses"),
    )

    planner = builder.AddSystem(Planner(plant, rng))
    builder.Connect(
        station.GetOutputPort("body_poses"), planner.GetInputPort("body_poses")
    )
    builder.Connect(
        x_bin_grasp_selector.get_output_port(),
        planner.GetInputPort("x_bin_grasp"),
    )
    builder.Connect(
        y_bin_grasp_selector.get_output_port(),
        planner.GetInputPort("y_bin_grasp"),
    )
    builder.Connect(
        station.GetOutputPort("wsg.state_measured"),
        planner.GetInputPort("wsg_state"),
    )
    builder.Connect(
        station.GetOutputPort("iiwa.position_measured"),
        planner.GetInputPort("iiwa_position"),
    )

    robot = station.GetSubsystemByName("iiwa_controller_plant_pointer_system").get()
    diff_ik = AddIiwaDifferentialIK(builder, robot)
    builder.Connect(planner.GetOutputPort("X_WG"), diff_ik.get_input_port(0))
    builder.Connect(
        station.GetOutputPort("iiwa.state_estimated"),
        diff_ik.GetInputPort("robot_state"),
    )
    builder.Connect(
        planner.GetOutputPort("reset_diff_ik"),
        diff_ik.GetInputPort("use_robot_state"),
    )

    builder.Connect(
        planner.GetOutputPort("wsg_position"),
        station.GetInputPort("wsg.position"),
    )

    switch = builder.AddSystem(PortSwitch(7))
    builder.Connect(diff_ik.get_output_port(), switch.DeclareInputPort("diff_ik"))
    builder.Connect(
        planner.GetOutputPort("iiwa_position_command"),
        switch.DeclareInputPort("position"),
    )
    builder.Connect(switch.get_output_port(), station.GetInputPort("iiwa.position"))
    builder.Connect(
        planner.GetOutputPort("control_mode"),
        switch.get_port_selector_input_port(),
    )

    builder.AddSystem(StopButton(meshcat))

    diagram = builder.Build()
    simulator = Simulator(diagram)
    context = simulator.get_context()

    plant_context = plant.GetMyMutableContextFromRoot(context)
    z = 0.2
    for body_index in plant.GetFloatingBaseBodies():
        tf = RigidTransform(
            UniformlyRandomRotationMatrix(generator),
            [rng.uniform(0.35, 0.65), rng.uniform(-0.12, 0.28), z],
        )
        plant.SetFreeBodyPose(plant_context, plant.get_body(body_index), tf)
        z += 0.1

    simulator.AdvanceTo(0.1)
    meshcat.Flush()

    simulator.set_target_realtime_rate(0.0)
    simulator.AdvanceTo(np.inf)
