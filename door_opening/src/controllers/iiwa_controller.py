from pydrake.multibody.plant import MultibodyPlant
from manipulation.station import LoadScenario, MakeMultibodyPlant
from manipulation.utils import FindResource

def CreateIiwaPlant() -> tuple[MultibodyPlant, list[int]]:
    """creates plant that includes only the robot and gripper, used for controllers."""
    scenario = LoadScenario(filename=FindResource("models/cupboard.scenario.yaml"))
    plant_robot = MakeMultibodyPlant(
        scenario=scenario, model_instance_names=["iiwa", "wsg"]
    )

    link_frame_indices = []
    for i in range(8):
        link_frame_indices.append(
            plant_robot.GetFrameByName("iiwa_link_" + str(i)).index()
        )

    return plant_robot, link_frame_indices