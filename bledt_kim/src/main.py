import argparse

from pydrake.geometry import StartMeshcat, MeshcatVisualizer
from pydrake.math import RigidTransform
from pydrake.systems.analysis import Simulator

from robot import ROBOTS
from simulation import build_system


def parse_args():
    parser = argparse.ArgumentParser(description="Robot walking simulation")
    parser.add_argument(
        "--robot",
        choices=ROBOTS.keys(),
        default="mini_cheetah",
        help="Robot to simulate (choices: mini_cheetah)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=5.0,
        help="Simulation duration in seconds (default: 5.0)",
    )
    parser.add_argument(
        "--v-des",
        type=float,
        default=0.0,
        help="Desired forward velocity m/s (0=march in place, 0.2=trot) - used when --teleop off",
    )
    parser.add_argument(
        "--teleop",
        choices=["off", "keyboard", "joystick", "both", "auto"],
        default="off",
        help="Teleop: off=constant v_des, keyboard=WASD+Q/E, joystick=Xbox Razer (left Y=vx, left X=vy, right X=wz), both=joint, auto=joy else kbd",
    )
    parser.add_argument("--max-vx", type=float, default=0.6, help="Teleop max vx m/s")
    parser.add_argument("--max-vy", type=float, default=0.4, help="Teleop max vy m/s (strafe)")
    parser.add_argument("--max-wz", type=float, default=1.0, help="Teleop max wz rad/s (turn)")
    parser.add_argument("--joystick-id", type=int, default=0, help="pygame joystick id for Xbox controller")
    return parser.parse_args()


def main():
    args = parse_args()
    robot_class = ROBOTS[args.robot]

    meshcat = StartMeshcat()
    builder, plant, scene_graph, model, robot = build_system(
        robot_class,
        v_des=args.v_des,
        teleop_mode=args.teleop,
        teleop_max_vx=args.max_vx,
        teleop_max_vy=args.max_vy,
        teleop_max_wz=args.max_wz,
        joystick_id=args.joystick_id,
    )

    # Visualizer Setup
    MeshcatVisualizer.AddToBuilder(builder, scene_graph, meshcat)

    # Build Diagram & Create Context
    diagram = builder.Build()
    context = diagram.CreateDefaultContext()
    plant_context = plant.GetMyMutableContextFromRoot(context)

    # Initial pose: nominal config puts feet on ground
    q_nominal = robot_class.Q_NOMINAL
    body = plant.GetBodyByName(robot_class.BASE_BODY_NAME, model)
    plant.SetFreeBodyPose(
        plant_context, body, RigidTransform([0, 0, robot_class.BASE_HEIGHT])
    )
    robot.set_positions(plant_context, q_nominal)

    # Run Simulation
    simulator = Simulator(diagram, context)
    meshcat.StartRecording()

    print(f"Robot: {args.robot} | v_des: {args.v_des} | teleop: {args.teleop}")
    if args.teleop != "off":
        print(f"Teleop {args.teleop}: max_vx={args.max_vx} max_vy={args.max_vy} max_wz={args.max_wz} | joy_id={args.joystick_id}")
        print("  joystick: left Y=forward, left X=strafe, right X=turn | keyboard: W/S forward, A/D strafe, Q/E turn")
    print(f"Meshcat URL: {meshcat.web_url()}")
    simulator.AdvanceTo(args.duration)

    meshcat.PublishRecording()
    input("\nPress Enter to exit...")


if __name__ == "__main__":
    main()
