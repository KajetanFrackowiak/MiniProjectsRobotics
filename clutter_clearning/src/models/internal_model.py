from pydrake.systems.framework import DiagramBuilder
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph

from manipulation.utils import ConfigureParser


def make_internal_model():
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.001)
    parser = Parser(plant)
    ConfigureParser(parser)
    path = parser.package_map().ResolveUrl(
        "package://manipulation/clutter_planning.dmd.yaml"
    )
    parser.AddModels(path)
    plant.Finalize()
    diagram = builder.Build()
    return diagram
