import numpy as np
from pydrake.systems.framework import LeafSystem


class FootstepPlanner(LeafSystem):
    """Raibert heuristic footstep planner (WBC_MPC.md sec 3).

    r_i^cmd = p_shoulder,i + p_symmetry + p_centrifugal
    p_shoulder,i = p_base + R_z(yaw) * l_i
    p_symmetry = t_stance/2 * v + k * (v - v_cmd)
    p_centrifugal = 0.5*sqrt(h/g) * (v x omega_cmd)

    Inputs:
      base_pose (6): [x,y,z,roll,pitch,yaw]
      base_twist (6): [wx,wy,wz, vx,vy,vz] (world)
      cmd_vel (3): [vx_cmd, vy_cmd, wz_cmd]
    Output:
      foot_offset (2): [offset_x, offset_y] = p_sym + p_cent (shared)
      foot_placement (12): world r_i for 4 feet (z=0) = shoulder + offset
    For now offset is shared across feet (straight trot); per-foot
    shoulder differs due to hip offsets.
    """

    def __init__(
        self,
        hip_offsets,
        period=0.5,
        duty_factor=0.65,
        k_gain=0.03,
        g=9.81,
    ):
        super().__init__()
        self.hip_offsets = [np.array(o) for o in hip_offsets]
        self.period = period
        self.duty_factor = duty_factor
        self.k_gain = k_gain
        self.g = g

        self.DeclareVectorInputPort("base_pose", 6)
        self.DeclareVectorInputPort("base_twist", 6)
        self.DeclareVectorInputPort("cmd_vel", 3)

        self.DeclareVectorOutputPort("foot_offset", 2, self.CalcOffset)
        self.DeclareVectorOutputPort("foot_placement", 12, self.CalcPlacement)

    def _compute_offset(self, context):
        base_pose = self.get_input_port(0).Eval(context)
        base_twist = self.get_input_port(1).Eval(context)
        cmd_vel = self.get_input_port(2).Eval(context)

        yaw = base_pose[5]
        c, s = np.cos(yaw), np.sin(yaw)
        R_z = np.array([[c, -s], [s, c]])

        # base horizontal velocity in body frame
        v_world_xy = np.array([base_twist[3], base_twist[4]])
        v_body_xy = R_z.T @ v_world_xy
        v_cmd_xy = np.array([cmd_vel[0], cmd_vel[1]])
        wz_cmd = float(cmd_vel[2])

        h = float(base_pose[2])
        if h < 0.05:
            h = 0.285  # fallback nominal

        t_stance = self.duty_factor * self.period

        # Symmetry term Raibert in body frame
        p_sym_x = t_stance * 0.5 * v_body_xy[0] + self.k_gain * (v_body_xy[0] - v_cmd_xy[0])
        p_sym_y = t_stance * 0.5 * v_body_xy[1] + 0.035 * (v_body_xy[1] - v_cmd_xy[1])

        # Centrifugal term (inverted pendulum) in body frame
        sqrt_hg = np.sqrt(h / self.g) if h > 0 else 0.3
        p_cent_y = 0.5 * sqrt_hg * (-v_body_xy[0] * wz_cmd)

        off_x = p_sym_x
        # Bound lateral stepping offset to +/- 3.5 cm to prevent roll drift while keeping equal stance width
        off_y = float(np.clip(p_sym_y + p_cent_y, -0.035, 0.035))

        offset_body = np.array([off_x, off_y])
        return offset_body, base_pose, v_world_xy

    def CalcOffset(self, context, output):
        offset, _, _ = self._compute_offset(context)
        output.SetFromVector(offset)

    def CalcPlacement(self, context, output):
        offset_body, base_pose, _ = self._compute_offset(context)
        yaw = base_pose[5]
        c, s = np.cos(yaw), np.sin(yaw)
        R_z = np.array([[c, -s], [s, c]])
        base_xy = base_pose[:2]
        p = np.zeros(12)
        for i, hip in enumerate(self.hip_offsets):
            shoulder = base_xy + R_z @ hip
            r_xy = shoulder + R_z @ offset_body
            p[3 * i : 3 * i + 3] = [r_xy[0], r_xy[1], 0.0]
        output.SetFromVector(p)
