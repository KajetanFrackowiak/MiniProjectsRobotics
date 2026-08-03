from manipulation.scenarios import AddMultibodyTriad, SetColor
from pydrake.geometry import MeshcatVisualizer, MeshcatVisualizerParams, SceneGraph
from pydrake.multibody.plant import MultibodyPlant
from pydrake.systems.framework import DiagramBuilder, OutputPort
from pydrake.systems.rendering import MultibodyPositionToGeometryPose

from utils.peg_geometry import add_planar_peg_to_plant


def add_setpoint_visualization(
    builder: DiagramBuilder,
    meshcat: MeshcatVisualizer,
    pos_traj_port: OutputPort,
    peg_frame_offset: float = 0.0,
) -> None:
    # Use the controller plant to visualize the set point geometry.
    controller_scene_graph = builder.AddSystem(SceneGraph())
    controller_plant = MultibodyPlant(time_step=0.005)
    controller_plant.RegisterAsSourceForSceneGraph(controller_scene_graph)
    add_planar_peg_to_plant(controller_plant, peg_frame_offset)
    controller_plant.Finalize()
    SetColor(
        controller_scene_graph,
        color=[1.0, 0.0, 0.0, 0.2],
        source_id=controller_plant.get_source_id(),
    )
    AddMultibodyTriad(
        controller_plant.GetFrameByName("peg_tip_frame"), controller_scene_graph
    )
    controller_vis = MeshcatVisualizer.AddToBuilder(
        builder,
        controller_scene_graph,
        meshcat,
        MeshcatVisualizerParams(prefix="controller"),
    )
    controller_vis.set_name("controller meshcat")
    positions_to_poses = builder.AddSystem(
        MultibodyPositionToGeometryPose(controller_plant)
    )
    builder.Connect(
        positions_to_poses.get_output_port(),
        controller_scene_graph.get_source_pose_port(controller_plant.get_source_id()),
    )
    builder.Connect(pos_traj_port, positions_to_poses.get_input_port())
