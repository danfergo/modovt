from old_src.modules.module import Module


class MotorModule(Module):

    def __init__(self):
        super(MotorModule, self).__init__()
        # self.motor_encoder = None
        # self.motor_decoder = None
        # self.motor_dynamics = motor_decoder(motor_encoder + action)

    def on_goal(self, *args):
        pass

    def on_observation(self, arm_joints, gripper_aperture):
        # encode raw observation
        # e_motor = self.motor_encoder(enouch2, arm_joints, gripper_aperture)

        # sample action according to current goal

        self.emit_observation(e_motor)


    def on_feedback(self, *args):
        pass