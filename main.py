import gym
import numpy as np
from skimage.transform import resize


def crop( img):
    img_cropped = np.zeros((192, 148))
    img_cropped[:, :] = img[3:195, 6:154]
    return img_cropped

def to_grayscale(img):
    return np.mean(img, axis=2)


def to_int(img):
    return img.astype(np.uint8)


def to_float(img):
    return img.astype(np.float)


def downsample(img):
    return img[::2, ::2]


def preprocess(img):
    return to_int(downsample(crop(to_grayscale(img))))


env = gym.make('Breakout-v0')
Q = np.zeros([14208, env.action_space.n])

eta = .628
gma = .9
epis = 5000
rev_list = []

for i in range(epis):
    s = env.reset()
    s_none, _, _, _ = env.step(0)
    s = preprocess(s)
    s_none = preprocess(s_none)
    s = s.ravel()
    s_none = s_none.ravel()
    s = np.concatenate((s, s_none), axis=0)
    a = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
    rAll = 0
    d = False
    while not d:
        env.render()
        a1 = np.argmax(Q[s, :] + np.random.randn(1, env.action_space.n) * (1. / (i + 1)))
        s1, r, d, _ = env.step(a)
        env.render()
        s1_none, _, _, _ = env.step(0)
        s1 = preprocess(s1)
        s1 = s1.ravel()
        s1_none = preprocess(s1_none)
        s1_none = s1_none.ravel()
        s1 = np.concatenate((s1, s1_none), axis=0)
        # Sarsa
        Q[s, a] = Q[s, a] + eta * (r + gma * Q[s1, a1] - Q[s, a])
        rAll += r
        s = s1
        a = a1
    rev_list.append(rAll)
    env.render()
    if i >0 and i % 100 == 0:
        print("Reward Sum on "+str(i)+" episodes " + str(sum(rev_list) / i))
        print("Reward Max on " + str(i) + " episodes " + str(max(rev_list)))

print("Reward Sum on all episodes " + str(sum(rev_list) / epis))
print("Final Values Q-Table")
print(Q)
