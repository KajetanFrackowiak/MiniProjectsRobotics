import numpy as np
from pydrake.systems.framework import LeafSystem

class GaitScheduler(LeafSystem):
    def __init__(self, period=0.5, duty_factor=0.5):
        super().__init__()
        self.period = period
        self.duty_factor = duty_factor

        # Trot [FR, FL, HR, HL]
        self.phase_offset = np.array([0.0, 0.5, 0.5, 0.0])

        # Planned state s_phi (0 = swing, 1 = stance) for 4 legs
        self.DeclareVectorOutputPort("s_phi", 4, self.CalcSchedulerState)
        # Subphase phi \in [0, 1) for 4 legs
        self.DeclareVectorOutputPort("phi", 4, self.CalcSubphase)


    def CalcSubphase(self, context, output):
        t = context.get_time()
        phi = np.mod(t / self.period + self.phase_offset, 1.0)
        output.SetFromVector(phi)

    def CalcSchedulerState(self, context, output):
        t = context.get_time()
        phi = np.mod(t / self.period + self.phase_offset, 1.0)
        # s_phi = 1 when the phase is within the support band, 0 during swing
        s_phi = (phi < self.duty_factor).astype(np.float64)
        output.SetFromVector(s_phi)



    