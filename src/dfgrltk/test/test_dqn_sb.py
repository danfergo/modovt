import gym

from stable_baselines3 import DQN, SAC

from src.dfgrltk.test.shared.sb_gym_adapter import SBGymAdapter

# env = SBGymAdapter('CartPole-v1')
# env = SBGymAdapter('LunarLander-v2')
env = SBGymAdapter('Pendulum-v1')

# model = DQN("MlpPolicy", env, verbose=1)
# model.learn(total_timesteps=10000, log_interval=4)

model = SAC("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=4)

# model.save("dqn_cartpole")
#
# del model  # remove to demonstrate saving and loading
#
# model = DQN.load("dqn_cartpole")
#
# obs = env.reset()
# while True:
#     action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()
#     if done:
#         obs = env.reset()