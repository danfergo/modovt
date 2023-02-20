@behaviour(
    components=[
        ArmComponent,
    ],
    defaults={
        'observe': lambda self: (self.gripper.at(), self.gripper1.at(), self.gripper2.at())
    }

)
class GripperBehaviour:

    def __init__(self, injector):
        self.gripper = injector.get('gripper')
        self.geltip1 = injector.get('geltip1')
        self.geltip2 = injector.get('geltip2')

        self.e_goal = None
        self.state = None

    # def value(self, state, observation):
    #     return self.value_fn(statea, observation)

    def get_state_representation(self, observation):
        return self.encoder(observation)

    def max_depth(self, geltip):
        return 0

    def i_value(self):
        max_protrusion = max(self.max_depth(self.geltip1), self.max_depth(self.geltip2))
        return 0.001 - max(0.001, max_protrusion)

    def i_policy(self):
        if self.i_value() <= 0.002:
            return 2  # move back gripper
        elif self.goal:
            # elseif,  there's a goal and we are not at the goal
            return 1  # carry out our mission
        else:
            return 0  # pass data upstream

    def e_policy(self, state, goal):
        pass

    def on_observation(self, observation):
        self.observation = observation
        self.state = self.encoder(*observation)

        i_action = self.instrinsic_policy()

        if i_action == 2:
            # there's nothing to do, push the data upstream
            self.emit_goal(self.encoded_arm_state_next)
        elif i_action == 1:
            # i think I know what i'm doing .. let's try to carry out our missionś
            self.on_goal(self.e_goal)
        elif i_action == 0:
            self.gripper.move_rel(-0.001)

        # push into memory
        # observation, action, current goal

    def on_goal(self, goal):
        self.e_goal = goal

        action = self.e_policy(self.state, self.e_goal)
        self.next_expected_state = self.dynamics_model(self.state, action)
        self.next_expected_observation = self.decoder(self.next_expected_state)
        self.gripper.move_rel(self.next_expected_observation[0] - self.observation[0])


    def on_learn(self, memory):
        state = self.memory.state
        next_state = self.memory.next_state
        observation = self.decode(state)
        next_observation = self.decode(next_state)

        # train autoencoder
        fit_to_sample(self.ae, observation, next_observation)

        fit_to_sample

        # {id: xxx, observation: , goals: ,  set_goal, next: , prev: }

        # update encoder / decoders
        # if en
        pass

    # compute gradient's
    # update Q(s,a)
    # update V(s, a)
    # update policy
