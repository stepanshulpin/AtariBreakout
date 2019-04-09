import gym
env = gym.make('Breakout-v0')
frame = env.reset()
env.render()

is_done = False
while not is_done:
  frame, reward, is_done, _ = env.step(env.action_space.sample())
env.render()