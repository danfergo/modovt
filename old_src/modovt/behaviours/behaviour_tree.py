from yarok import Platform, ConfigBlock, Injector, behaviour


@behaviour(
    defaults={

    }
)
class BehaviourTree:

    def __init__(self, pl: Platform, injector: Injector, config: ConfigBlock):
        # init scripts
        self.behaviours = [bh(injector) for bh in config['scripts']]

        self.behaviours_levels = [[] + self.behaviours]
        self.top_behaviours = self.behaviours_levels[len(self.behaviours) - 1]
        self.top_behaviours = self.behaviours_levels[0]
        level = 0
        # whileC

        self.mode = config['mode']  # awake, sleep, eval
        self.eval_options = config['eval_options'] if 'eval_options' in config else None

        self.prev_memory = None
        self.current_memory = None

    def get_state_representation(self, observation):
        return [bh.get_state_representation(observation) for bh in self.top_behaviours]

    def feed_observations_upstream(self):
        pass

    def on_update(self):
        if self.mode == 'awake':
            current_memory = {'id': self.current_memory['id'] + 1 if self.current_memory is not None else 1}
            self.prev_memory['next'] = current_memory
            self.prev_memory = self.current_memory
            self.current_memory = {
                'prev': self.current_memory
            }

            # iterate through all leaf scripts and dispatch data.
            observations = [bh.observation() for bh in self.behaviours]

            encoded_observations = [
                self.behaviours[i].encode(observations[i])
                for i in range(len(self.behaviours))
            ]
        elif self.mode == 'sleep':
            observations = [bh.on_learn() for bh in self.behaviours]
        elif self.mode == 'eval':
            self.goal = self.config

            self.encoded_goal = self.get_state_representation(self.goal)

            for n in range(self.eval_options['n_trials']):

                for h in range(self.eval_options['horizon']):
                    pass
                    # propagate observations upstream
                    # self.propagate

               # assert after forward pass, the observed state is equals to the goal
               observations = [bh.observation() for bh in self.behaviours]

