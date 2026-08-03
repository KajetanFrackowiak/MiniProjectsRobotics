from manipulation import ConfigureParser
from pydrake.math import RigidTransform
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import MultibodyPlant
from pydrake.multibody.tree import (
    FixedOffsetFrame,
    ModelInstanceIndex,
    PrismaticJoint,
    RevoluteJoint,
    SpatialInertia,
    UnitInertia,
)

from utils.assets import HOLE_SDF_PATH, PEG_URDF_PATH, TABLE_SDF_PATH


def add_planar_peg_to_plant(
    plant: MultibodyPlant, peg_frame_offset: float = 0.0
) -> ModelInstanceIndex:

    parser = Parser(plant)
    ConfigureParser(parser)

    peg = parser.AddModels(str(PEG_URDF_PATH))[0]

    # Since in drake one joint always connects two frames (which in general
    # belongs to two rigid bodies), so we cannot simply create:
    # world -> prismatic_x -> prismatic_z -> revolute_theta -> peg
    # and we have to connect them by fictitious rigid bodies (with zero mass
    # and inertia):
    # world -> prismatic_x -> false_body1 -> prismatic_z -> false_body2
    #   -> revolute_theta -> peg
    _peg_false_body1 = plant.AddRigidBody(
        "false_body1",
        peg,
        # mass = 0, center of mass w.r.t body frame = [0, 0, 0], unit inertia
        # = [0, 0, 0] (Ixx, Iyy, Izz)
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )
    _peg_false_body2 = plant.AddRigidBody(
        "false_body2",
        peg,
        SpatialInertia(0, [0, 0, 0], UnitInertia(0, 0, 0)),
    )

    # world -> prismatic_x -> false_body_1
    peg_x = plant.AddJoint(
        PrismaticJoint(
            "peg_x",
            plant.world_frame(),
            plant.GetFrameByName("false_body1"),
            [1, 0, 0],
            -10,
            10,
        )
    )
    plant.AddJointActuator("peg_x", peg_x)

    # false_body_1 -> prismatic_z -> false_body_2
    peg_z = plant.AddJoint(
        PrismaticJoint(
            "peg_z",
            plant.GetFrameByName("false_body1"),
            plant.GetFrameByName("false_body2"),
            [0, 0, 1],
            -10,
            10,
        )
    )
    peg_z.set_default_translation(0.0)
    plant.AddJointActuator("peg_z", peg_z)

    peg_frame = plant.AddFrame(
        FixedOffsetFrame(
            "peg_tip_frame",
            plant.GetFrameByName("peg_body_link", peg),
            RigidTransform([0, 0, peg_frame_offset]),
        )
    )
    # false_body_2 -> revolute_theta -> peg
    peg_theta = plant.AddJoint(
        RevoluteJoint(
            "peg_theta",
            plant.GetFrameByName("false_body2"),
            peg_frame,
            [0, 1, 0],
            -10,
            10,
        )
    )
    plant.AddJointActuator("peg_theta", peg_theta)

    return peg


def add_table_to_plant(plant: MultibodyPlant) -> ModelInstanceIndex:

    parser = Parser(plant)
    ConfigureParser(parser)

    table = parser.AddModels(str(TABLE_SDF_PATH))[0]

    plant.WeldFrames(
        plant.world_frame(),
        plant.GetFrameByName("link", table),
        RigidTransform([0.5, 0, -0.7645]),
    )
    return table


def add_planar_hole_to_plant(
    plant: MultibodyPlant, table: ModelInstanceIndex
) -> ModelInstanceIndex:

    parser = Parser(plant)
    ConfigureParser(parser)

    hole = parser.AddModels(str(HOLE_SDF_PATH))[0]

    plant.WeldFrames(
        plant.GetFrameByName("link", table),
        plant.GetFrameByName("hole_chamfered_body_link", hole),
        RigidTransform([0, 0.125, 0.7645]),
    )
    return hole
