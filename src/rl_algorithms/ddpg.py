import torch

n_actions = 6
lr = 0.01
discount_factor = 0.95
T = 10


class Reinforce:

    def __init__(self, nn, optim=None):
        self.nn = nn
        self.optim = optim or torch.optim.Adam(nn.parameters(), lr=lr)

    def eval_return(self, t, rewards):
        return sum([(discount_factor ** k) * r for k, r in enumerate(rewards[t:])])

    def run(self, get_observation, get_reward, take_action):
        while True:
            # episode buffers
            episode_actions = []
            episode_observations = []
            episode_rewards = []

            print('exploring')
            # generate an episode
            for t in range(T):
                # get observation from the environment
                o = get_observation()
                o = torch.tensor(o, dtype=torch.float)
                o = torch.unsqueeze(o, 0)  # make a batch of one.

                # calculate the reward
                r = get_reward()

                # sample action, given the observation, following the policy
                pi_st = torch.distributions.Categorical(probs=self.nn(o))
                a = pi_st.sample().item()

                # st, at, rt, into the buffers
                episode_actions.append(a)
                episode_observations.append(o)
                episode_rewards.append(r)

                # take action and advance the environment
                take_action(a)

            print('learning')
            returns = [self.eval_return(t, episode_rewards) for t in range(len(episode_rewards))]

            for st, at, gt in zip(episode_observations, episode_actions, returns):
                at = torch.tensor(at, dtype=torch.int)
                pi_st = torch.distributions.Categorical(probs=self.nn(st))

                log_prob = pi_st.log_prob(at)  # the log probability of this action being taken, i.e. log_πθ(st|at)

                loss = - log_prob * gt  # log_πθ(st|at) * g
                # the negative is used because we want to maximize the expected returns,
                # while the optimizer minimizes

                self.optim.zero_grad()  # zero the radient graph for every parameter x
                loss.backward()  # compute the gradients dloss/dx
                self.optim.step()  # performs the gradient update for every x i.e., x += -lr * x.grad
