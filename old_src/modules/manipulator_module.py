from old_src.modules.module import Module


class ManipulatorModule(Module):

    def __init__(self):
        super(ManipulatorModule, self).__init__()
        self.manipulator_encoder = None
        self.visual_cortex = None
        self.motor_cortex = None

    def encode_observation(self, g_image, g_joints, g_gripper, g_tactile):
        encoded_visual = self.visual_cortex.encode_observation(g_image)
        encoded_motor = self.motor_cortex.encode_observation(g_joints, g_gripper, g_tactile)
        return self.manipulator_encoder(encoded_visual, encoded_motor)

    def on_goal(self, encoded_g):
        # sample action
        self.motor_cortex.set_goal()

    def on_observation(self, encoded_visual, encoded_motor):
        e_observation_t = self.manipulator_encoder(encoded_visual, encoded_motor)

        # e_observation_t1 = e_observation_t + e_action

        # Q(s, a) = Q(e_observation_t, e_action)

        # sample an action
        self.motor_cortex(e_observation_t)

    def on_feedback(self, encoded_visual_rb, encoded_motor_rb, actions_rb):
        batch = sample(encoded_visual_rb, encoded_motor_rb)
        self.manipulator_encoder(batch)


        # g-SAC