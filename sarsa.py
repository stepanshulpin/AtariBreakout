import resource
import sys

import gym
import numpy as np
from utils import get_state, convert_to_direction

env = gym.make('Breakout-v0')

# state1 25-height of ball/2 + 1 37-width of ball/2 + 1 18-width/4
# state2 25 37 18
#Q = np.zeros([25, 73, 72, 3, 3, 3, env.action_space.n])
Q = np.load("Q_Table_SARSA.npy")
actions = range(env.action_space.n)
eta = .628
gma = .9
epis = 10000
rev_list = []
epsilon = 0.1

for i in range(epis):
    s = env.reset()
    s_none, _, _, _ = env.step(0)
    s = get_state(s)
    s_none = get_state(s_none)
    s_none = convert_to_direction(s, s_none)
    s = s + s_none
    a = np.argmax(Q[s[0], s[1], s[2], s[3], s[4], s[5], :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
    rAll = 0
    d = False
    while not d:
        env.render()
        s_next, r, d, _ = env.step(a)
        env.render()
        s_next_none, _, _, _ = env.step(0)
        s_next = get_state(s_next)
        s_next_none = get_state(s_next_none)
        s_next_none = convert_to_direction(s_next, s_next_none)
        s_next = s_next + s_next_none
        a1 = np.argmax(
            Q[s_next[0], s_next[1], s_next[2], s_next[3], s_next[4], s_next[5], :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        Q[s[0], s[1], s[2], s[3], s[4], s[5], a] = Q[s[0], s[1], s[2], s[3], s[4], s[5], a] + eta * (
                    r + gma * Q[s_next[0], s_next[1], s_next[2], s_next[3], s_next[4], s_next[5], a1] - Q[s[0], s[1], s[2], s[3], s[4], s[5], a])
        rAll += r
        s = s_next
        a = a1
    rev_list.append(rAll)
    env.render()
    if i > 0 and i % 100 == 0:
        print("Reward Sum on " + str(i) + " episodes " + str(sum(rev_list) / i))
        print("Reward Max on " + str(i) + " episodes " + str(max(rev_list)))

print("Reward Sum on all episodes " + str(sum(rev_list) / epis))
np.save("Q_Table_SARSA.npy",Q)
print("Final Values Q-Table")

