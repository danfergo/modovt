class Policy:
    DISCRETE_AS = 0
    AS_CONTINUOUS = 1

    def __init__(self, action_space=0):
        self.action_space = action_space

    def log_prob(self, observation, action):
        pass

    def __call__(self, observation):
        pass
