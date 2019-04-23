import resource
import sys
import time

import gym
import numpy as np
from utils import get_state, convert_to_direction

env = gym.make('Breakout-v0')

# state1 13-height of ball/2 + 1 37-width of ball/2 + 1 18-width/4
# state2 25 37 18
#Q = np.zeros([25, 73, 72, 3, 3, 3, env.action_space.n])
Q = np.load("Q_Table.npy")
#print(sys.getsizeof(Q))
actions = range(env.action_space.n)
eta = .628
gma = .9
epis = 10000
rev_list = []
epsilon = 0.1
# get_state_time =0
# get_max_time = 0
# start_time = time.clock()
for i in range(epis):
    s = env.reset()
    s_none, _, _, _ = env.step(0)
    # start_get_state = time.clock()
    s = get_state(s)
    # get_state_time += time.clock() - start_get_state
    # start_get_state = time.clock()
    s_none = get_state(s_none)
    # get_state_time += time.clock() - start_get_state
    s_none = convert_to_direction(s, s_none)
    s = s + s_none
    rAll = 0
    d = False
    while not d:
        env.render()
        # a = act(s, Q, env, epsilon, actions)
        # start_get_max = time.clock()
        a = np.argmax(
            Q[s[0], s[1], s[2], s[3], s[4], s[5], :] + np.random.randn(1, env.action_space.n) * (0.1 / (i + 1)))
        # get_max_time += time.clock() - start_get_max
        s_next, r, d, _ = env.step(a)
        env.render()
        s_next_none, _, _, _ = env.step(0)
        # start_get_state = time.clock()
        s_next = get_state(s_next)
        # get_state_time += time.clock() - start_get_state
        # start_get_state = time.clock()
        s_next_none = get_state(s_next_none)
        # get_state_time += time.clock() - start_get_state
        s_next_none = convert_to_direction(s_next, s_next_none)
        s_next = s_next + s_next_none
        # start_get_max = time.clock()
        Q[s[0], s[1], s[2], s[3], s[4], s[5], a] = (1 - eta) * Q[s[0], s[1], s[2], s[3], s[4], s[5], a] + eta * (
                r + gma * np.max(Q[s_next[0], s_next[1], s_next[2], s_next[3], s_next[4], s_next[5], :]))
        # get_max_time += time.clock() - start_get_max
        rAll += r
        s = s_next
    rev_list.append(rAll)
    env.render()
    # if i > 0 and i % 10 == 0:
    #     time_all = time.clock() - start_time
    #     print("get state time percent " + str(i) + " episodes " + str(100 * get_state_time / (time_all)))
    #     print("get max time percent " + str(i) + " episodes " + str(100 * get_max_time / (time_all)))
    if i > 0 and i % 100 == 0:
        print("Reward Sum on " + str(i) + " episodes " + str(sum(rev_list) / i))
        print("Reward Max on " + str(i) + " episodes " + str(max(rev_list)))

print("Reward Sum on all episodes " + str(sum(rev_list) / epis))
np.save("Q_Table.npy", Q)
print("Final Values Q-Table")
