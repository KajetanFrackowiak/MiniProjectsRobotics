import numpy as np

from pydrake.geometry import Box, CollisionFilterDeclaration, GeometrySet
from pydrake.multibody.parsing import Parser
from pydrake.multibody.plant import AddMultibodyPlantSceneGraph, CoulombFriction
from pydrake.multibody.tree import SpatialInertia, UnitInertia
from pydrake.math import RigidTransform
from pydrake.systems.framework import DiagramBuilder, LeafSystem
from pydrake.systems.primitives import Adder, ConstantVectorSource

from controllers.joint_pd import JointPDController
from controllers.wbic import WBICController
from controllers.state_extractor import JointStateExtractor
from controllers.mpc import MPCController
from estimation.contact_estimator import ContactEstimator
from estimation.state_estimator import StateEstimator
from locomotion.event_fsm import EventFSM
from locomotion.gait_scheduler import GaitScheduler
from locomotion.trajectory_generator import LegTrajectoryGenerator
from locomotion.footstep_planner import FootstepPlanner


class FootPositionExtractor(LeafSystem):
    """Computes actual toe positions in world frame for smooth liftoff latching."""

    def __init__(self, plant):
        super().__init__()
        self.plant = plant
        self.plant_context = plant.CreateDefaultContext()
        self.toe_frames = [
            plant.GetFrameByName("toe_fr"),
            plant.GetFrameByName("toe_fl"),
            plant.GetFrameByName("toe_hr"),
            plant.GetFrameByName("toe_hl"),
        ]
        self.DeclareVectorInputPort(
            "x_full", plant.num_positions() + plant.num_velocities()
        )
        self.DeclareVectorOutputPort("foot_p_act", 12, self.CalcFootPositions)

    def CalcFootPositions(self, context, output):
        x = self.get_input_port(0).Eval(context)
        nq = self.plant.num_positions()
        self.plant.SetPositions(self.plant_context, x[:nq])
        self.plant.SetVelocities(self.plant_context, x[nq:])
        p = np.zeros(12)
        for i, f in enumerate(self.toe_frames):
            pos = self.plant.CalcPointsPositions(
                self.plant_context, f, np.zeros((3, 1)), self.plant.world_frame()
            ).ravel()
            p[3 * i : 3 * i + 3] = pos
        output.SetFromVector(p)


class TorqueLimiter(LeafSystem):
    """Clip summed torque to actuator limits."""

    def __init__(self, num_joints, limit):
        super().__init__()
        self.limit = float(limit)
        self.DeclareVectorInputPort("torque_in", num_joints)
        self.DeclareVectorOutputPort("torque_out", num_joints, self.Calc)

    def Calc(self, context, output):
        tau = self.get_input_port(0).Eval(context)
        output.SetFromVector(np.clip(tau, -self.limit, self.limit))


def add_ground(plant):
    ground_model = plant.AddModelInstance("ground")
    ground = plant.AddRigidBody(
        "ground",
        ground_model,
        SpatialInertia(mass=1.0, p_PScm_E=np.zeros(3), G_SP_E=UnitInertia(1, 1, 1)),
    )
    plant.WeldFrames(plant.world_frame(), ground.body_frame())

    ground_shape = Box(100.0, 100.0, 0.1)
    plant.RegisterCollisionGeometry(
        ground,
        RigidTransform([0, 0, -0.05]),
        ground_shape,
        "ground_collision",
        CoulombFriction(0.8, 0.8),
    )
    plant.RegisterVisualGeometry(
        ground,
        RigidTransform([0, 0, -0.05]),
        ground_shape,
        "ground_visual",
        np.array([0.7, 0.45, 0.25, 1.0]),
    )


def build_system(
    robot_class,
    time_step=0.001,
    gait_period=None,
    gait_duty=None,
    v_des=0.0,
    teleop_mode="off",
    teleop_max_vx=0.6,
    teleop_max_vy=0.4,
    teleop_max_wz=1.0,
    joystick_id=0,
):
    """Build diagram (Kim et al. 2019 / Bledt et al. 2018)."""
    if gait_period is None:
        # Teleop: 0.5 trot; constant v_des > 0.25 -> 0.4
        if teleop_mode != "off":
            gait_period = 0.5
        else:
            gait_period = 0.4 if abs(v_des) > 0.25 else 0.5
    if gait_duty is None:
        gait_duty = 0.5
    step_h = 0.05 if teleop_mode != "off" else (0.04 if abs(v_des) < 1e-9 else 0.05)

    builder = DiagramBuilder()

    # Plant & scene graph
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=time_step)
    parser = Parser(plant)
    model = parser.AddModels(str(robot_class.URDF_PATH))[0]
    robot = robot_class(plant, model)

    for joint in robot.joints:
        plant.AddJointActuator(f"{joint.name()}_actuator", joint)

    add_ground(plant)
    plant.Finalize()

    # FIX: knees-back puts front thighs inside body mesh (~17 mm penetration).
    # This is a mesh artifact - Mini Cheetah URDF was designed for X-shape.
    # Deactivate body collision so the robot can trot without spurious
    # self-contact forces. Body never needs to collide with ground at
    # 0.285m height.
    body = plant.GetBodyByName(robot_class.BASE_BODY_NAME)
    body_fid = plant.GetBodyFrameIdOrThrow(body.index())
    scene_graph.collision_filter_manager().Deactivate(GeometrySet(body_fid))
    # Exclude leg-leg self collisions
    gs_thigh = GeometrySet()
    for name in ["thigh_fr", "thigh_fl", "thigh_hr", "thigh_hl"]:
        b = plant.GetBodyByName(name)
        gs_thigh.Add(plant.GetBodyFrameIdOrThrow(b.index()))
    scene_graph.collision_filter_manager().Apply(
        CollisionFilterDeclaration().ExcludeWithin(gs_thigh)
    )

    # Register systems
    state_extractor = builder.AddSystem(
        JointStateExtractor(
            num_positions=plant.num_positions(),
            num_velocities=plant.num_velocities(),
            num_joints=robot.num_joints,
        )
    )
    foot_extractor = builder.AddSystem(FootPositionExtractor(plant))

    gait_scheduler = builder.AddSystem(
        GaitScheduler(period=gait_period, duty_factor=gait_duty)
    )

    # Footstep planner (Raibert symmetry + centrifugal)
    hip_offsets = [
        np.array([0.19, -0.111]),
        np.array([0.19,  0.111]),
        np.array([-0.19, -0.111]),
        np.array([-0.19,  0.111]),
    ]
    footstep_planner = builder.AddSystem(
        FootstepPlanner(
            hip_offsets=hip_offsets,
            period=gait_period,
            duty_factor=gait_duty,
            k_gain=0.03,
        )
    )
    if teleop_mode != "off":
        from teleop.teleop import TeleopSystem

        cmd_vel_src = builder.AddSystem(
            TeleopSystem(
                mode=teleop_mode,
                max_vx=teleop_max_vx,
                max_vy=teleop_max_vy,
                max_wz=teleop_max_wz,
                joystick_id=joystick_id,
            )
        )
        # expose for debugging
        builder.teleop = cmd_vel_src
    else:
        cmd_vel_src = builder.AddSystem(
            ConstantVectorSource(np.array([float(v_des), 0.0, 0.0]))
        )

    # Trajectory generator (smooth liftoff-to-touchdown Bezier in world frame)
    traj_generator = builder.AddSystem(
        LegTrajectoryGenerator(
            step_height=step_h,
            duty_factor=gait_duty,
            period=gait_period,
            ground_clearance=robot_class.TOE_RADIUS,
        )
    )

    # Whole-Body Impulse Controller (Kim et al. 2019)
    wbic = builder.AddSystem(
        WBICController(
            plant=plant,
            robot_model=model,
            mu=0.6,
            tau_max=robot_class.MAX_TORQUE,
            q1_weight=1.0,  # Q1 = 1
            q2_weight=0.5,  # Q2 = 0.5
            kp_body_rot=100.0,
            kd_body_rot=10.0,
            kp_body_pos=100.0,
            kd_body_pos=10.0,
            kp_foot=100.0,
            kd_foot=10.0,
        )
    )

    # State Estimator (WBC_MPC.md:21)
    state_estimator = builder.AddSystem(StateEstimator(dt=0.001, cutoff_hz=15.0))

    # Contact Estimator (Bledt 2018)
    contact_estimator = builder.AddSystem(
        ContactEstimator(
            plant=plant,
            robot_model=model,
            dt=0.002,
            lambda_cut=15.0,
        )
    )

    # Event FSM (Bledt 2018 sec5) Early/Late handling
    event_fsm = builder.AddSystem(
        EventFSM(dt=0.001, t_delay=0.02, t_late_grace=0.04, v_extend=0.05)
    )

    # Low-level Joint PD (Kim et al. 2019: kp=3, kd=1 with stiffer abduction kp=15)
    kp_pd = np.array([3.0] * robot.num_joints)
    kd_pd = np.array([1.0] * robot.num_joints)
    for idx in [0, 3, 6, 9]:
        kp_pd[idx] = 15.0
        kd_pd[idx] = 1.0
    joint_pd = builder.AddSystem(
        JointPDController(
            num_joints=robot.num_joints,
            kp=kp_pd,
            kd=kd_pd,
            max_torque=None,
            use_v_des=True,
        )
    )
    torque_adder = builder.AddSystem(Adder(2, robot.num_joints))
    torque_limiter = builder.AddSystem(
        TorqueLimiter(robot.num_joints, robot_class.MAX_TORQUE)
    )

    # Diagram connections

    # Plant -> Extractor
    builder.Connect(plant.get_state_output_port(), state_extractor.get_input_port(0))
    builder.Connect(plant.get_state_output_port(), foot_extractor.get_input_port(0))

    # State Estimator
    builder.Connect(
        state_extractor.GetOutputPort("body_pose_rpy"),
        state_estimator.get_input_port(0),
    )
    builder.Connect(
        state_extractor.GetOutputPort("body_twist"), state_estimator.get_input_port(1)
    )

    # Footstep Planner
    builder.Connect(
        state_extractor.GetOutputPort("body_pose_rpy"),
        footstep_planner.get_input_port(0),
    )
    builder.Connect(
        state_extractor.GetOutputPort("body_twist"), footstep_planner.get_input_port(1)
    )
    builder.Connect(cmd_vel_src.get_output_port(), footstep_planner.get_input_port(2))

    # Gait Scheduler -> Trajectory Generator
    builder.Connect(gait_scheduler.get_output_port(0), traj_generator.get_input_port(0))
    builder.Connect(gait_scheduler.get_output_port(1), traj_generator.get_input_port(1))
    builder.Connect(
        footstep_planner.GetOutputPort("foot_placement"),
        traj_generator.get_input_port(2),
    )
    builder.Connect(
        foot_extractor.GetOutputPort("foot_p_act"), traj_generator.get_input_port(3)
    )

    # Event FSM wiring
    builder.Connect(
        gait_scheduler.get_output_port(0), event_fsm.get_input_port(0)
    )  # s_phi
    builder.Connect(
        contact_estimator.GetOutputPort("s_hat"), event_fsm.get_input_port(1)
    )
    builder.Connect(
        traj_generator.GetOutputPort("foot_pos_des"), event_fsm.get_input_port(2)
    )  # pos_des
    builder.Connect(
        traj_generator.GetOutputPort("foot_vel_des"), event_fsm.get_input_port(3)
    )  # vel_des

    # MPC (Kim et al. 2019)
    mpc = builder.AddSystem(
        MPCController(
            plant=plant,
            robot_model=model,
            period=gait_period,
            duty_factor=gait_duty,
            mu=0.6,
        )
    )
    builder.Connect(
        state_extractor.GetOutputPort("body_pose_rpy"), mpc.get_input_port(0)
    )
    builder.Connect(state_extractor.GetOutputPort("body_twist"), mpc.get_input_port(1))
    builder.Connect(event_fsm.GetOutputPort("s_eff"), mpc.get_input_port(2))
    builder.Connect(cmd_vel_src.get_output_port(), mpc.get_input_port(3))

    # WBIC Inputs
    builder.Connect(
        plant.get_state_output_port(), wbic.get_input_port(0)
    )  # x_full
    builder.Connect(mpc.get_output_port(0), wbic.get_input_port(1))  # f_mpc
    builder.Connect(event_fsm.GetOutputPort("s_eff"), wbic.get_input_port(2))  # s_eff
    builder.Connect(
        event_fsm.GetOutputPort("foot_pos_mod"), wbic.get_input_port(3)
    )  # foot_pos_mod
    builder.Connect(
        event_fsm.GetOutputPort("foot_vel_mod"), wbic.get_input_port(4)
    )  # foot_vel_mod

    # Body pose target: constant nominal height and zero orientation
    body_pose_src = builder.AddSystem(
        ConstantVectorSource(
            np.array([0.0, 0.0, robot_class.BASE_HEIGHT, 0.0, 0.0, 0.0])
        )
    )
    builder.Connect(
        body_pose_src.get_output_port(), wbic.get_input_port(5)
    )  # body_pose_des
    builder.Connect(cmd_vel_src.get_output_port(), wbic.get_input_port(6))  # cmd_vel

    # WBIC q_cmd/v_cmd -> Joint PD (feedback) + WBIC tau (feedforward) -> Plant
    builder.Connect(wbic.GetOutputPort("q_cmd"), joint_pd.get_input_port(0))
    builder.Connect(
        state_extractor.GetOutputPort("q_joints"), joint_pd.get_input_port(1)
    )
    builder.Connect(
        state_extractor.GetOutputPort("v_joints"), joint_pd.get_input_port(2)
    )
    builder.Connect(wbic.GetOutputPort("v_cmd"), joint_pd.get_input_port(3))

    builder.Connect(wbic.GetOutputPort("torque"), torque_adder.get_input_port(0))
    builder.Connect(joint_pd.GetOutputPort("torque"), torque_adder.get_input_port(1))
    builder.Connect(torque_adder.get_output_port(), torque_limiter.get_input_port(0))
    builder.Connect(torque_limiter.get_output_port(), plant.get_actuation_input_port())

    # Keep reference for debugging/tests
    builder.contact_estimator = contact_estimator
    builder.state_estimator = state_estimator
    builder.event_fsm = event_fsm
    builder.mpc = mpc
    builder.wbic = wbic
    builder.torque_limiter = torque_limiter
    builder.gait_scheduler = gait_scheduler
    builder.traj_generator = traj_generator
    builder.footstep_planner = footstep_planner
    builder.state_extractor = state_extractor
    builder.joint_pd = joint_pd

    # Contact Estimator wiring
    builder.Connect(
        plant.get_state_output_port(), contact_estimator.get_input_port(0)
    )
    builder.Connect(
        gait_scheduler.get_output_port(0), contact_estimator.get_input_port(1)
    )
    builder.Connect(
        gait_scheduler.get_output_port(1), contact_estimator.get_input_port(2)
    )
    builder.Connect(
        torque_limiter.get_output_port(), contact_estimator.get_input_port(3)
    )

    return builder, plant, scene_graph, model, robot
