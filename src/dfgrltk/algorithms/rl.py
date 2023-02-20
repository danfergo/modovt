from src.dfgrltk.replay_buffer import ReplayBuffer


class RL:

    def __init__(self, rb: ReplayBuffer, lr=0.005, gamma=0.9999):
        self.rb = rb
        self.lr = lr
        self.gamma = gamma

    def sdr(self, t_start):
        t_final = self.rb.episode_end_of(t_start)
        return sum([(self.gamma ** (t - t_start)) * self.rb.all('reward')[t] for t in range(t_start, t_final)])

    def update(self):
        """
            performs one learning update
        """
        # sample batch from memory,
        # compute gradients / errors.
        # update the policy, following the gradients.
        pass

    def online_learn(self, **kwargs):
        pass
