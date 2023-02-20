#
# from src.dfgrltk.algorithms.reinforce import Reinforce
# from src.dfgrltk.policies.random import RandomPolicy
# from src.dfgrltk.replay_buffer import Memory
#
#
# class Agent:
#
#     def __init__(self, path):
#         # self.memory = Memory.load_from_path(path)
#
#     def wake_up(self, mode='learn'):
#         self.mode = 'learn'
#
#     def sleep(self):
#         pass
#
#
# class AgentBehaviour(Agent):
#
#     def __init__(self):
#         mode = 'learn'  # 'eval'
#         memory_path = ''
#         behaviour = RandomPolicy()
#         target = DRLPolicy(
#             algo=Reinforce(
#
#             ),
#             model=model(self.path),
#             task=RewardBasedTask(
#
#                 reward=self.reward
#             )
#         )
#         super(AgentBehaviour).__init__(memory_path)
#
#         # self.rl = rl
#         # self.memory = memory
#         # self.n_episodes = 1000
#         # self.episode_length = 1000
#         # self.learning_steps = 1000
#
#     # @staticmethod
#     # def from_path(mem_path, algorithm_name=None):
#     #     m = Memory.load_from_path(mem_path)
#     #
#     #     if algorithm_name is None:
#     #         algorithm_name = m.meta()['rl_algorith']
#     #
#     #     # algorithms[algorithm_name]
#     #     #
#     #     # a = Agent(m, )
#     #     # return a
#
#     def reward(self):
#         pass
#
#     def act(self):
#         for i in range(self.n_episodes):
#             for j in range(self.episode_length):
#
#         # observation, reward = self.observe()
#         # action = self.rl.policy(observation)
#         # if action:
#         #     self.env
#         #
#         # if j > 0:
#         #
#         # previous_reward = reward
#
#     def on_observation(self, observation):
#         gripper_o = observation['gripper']
#
#         gripper_a = self.behaviour.__call__(gripper_o)
#
#         self.memory.append({
#             'o': gripper_o,
#             'a': gripper_a,
#             'r': previous_reward
#         })
#
#         return {
#             'gripper': gripper_a
#         }
