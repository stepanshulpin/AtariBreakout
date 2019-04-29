import matplotlib.pyplot as plt
import numpy as np


def show(s):
    fig, axes = plt.subplots()
    axes.imshow(s)
    plt.show()


def to_grayscale(img):
    # n = img.shape[0
    # m = img.shape[1]
    # res_img = np.zeros([n, m], dtype=np.bool_)
    # for i in range(n):
    #     for j in range(m):
    #         pixel = img[i, j, :]
    #         if any(p > 0 for p in pixel):
    #             res_img[i, j] = 1
    return np.mean(img, axis=2)
    #return res_img


def downsample(img):
    return img[::4, ::2]


def preprocess(img):
    return to_grayscale(downsample(img))


def cut(img):
    img_cropped = np.zeros([34, 72], dtype=np.bool_)
    img_cropped[:, :] = img[15:49, 4:76]
    return img_cropped


def saw(img):
    ball = np.zeros([24, 72], dtype=np.bool_)
    plat = np.zeros([1, 72], dtype=np.bool_)
    block = np.zeros([9, 72], dtype=np.bool_)
    ball[:, :] = img[9:33, :]
    plat[:, :] = img[33:34, :]
    block[:, :] = img[0:9, :]
    return ball, plat, block


def plat_pos(img):
    l = list(img[0, :])
    pos = 0
    while pos < len(l):
        if all(p != 0 for p in l[pos:pos + 4]):
            break
        pos += 1
    return [pos]


def ball_pos(img):
    n = img.shape[0]
    m = img.shape[1]
    pos = [24, 72]
    for i in range(n):
        for j in range(m):
            if img[i, j] != 0:
                pos = [i, j]
                break
    return pos


def get_state(s):
    s_prep = preprocess(s)
    s_cut = cut(s_prep)
    s_ball, s_plat, s_blok = saw(s_cut)
    return ball_pos(s_ball) + plat_pos(s_plat)


def act(s, Q, env, epsilon, actions):
    if np.random.random() < epsilon:
        return env.action_space.sample()

    qvals = {a: Q[s[0], s[1], s[2], s[3], s[4], s[5], a] for a in actions}
    max_q = max(qvals.values())
    actions_with_max_q = [a for a, q in qvals.items() if q == max_q]
    return np.random.choice(actions_with_max_q)

def convert_to_direction(s, s_none):
    res = [0,0,0]
    res[0] = convert_helper(s[0] - s_none[0])
    res[1] = convert_helper(s[1] - s_none[1])
    res[2] = convert_helper(s[2] - s_none[2])
    return res


def convert_helper(d):
    if d < 0:
        return 0
    if d == 0:
        return 1
    if d > 0:
        return 2


def learn(Q, s, s_next, a, eta, gma, r):
    Q[s[0], s[1], s[2], s[3], s[4], s[5], a] = \
        (1 - eta) * Q[s[0], s[1], s[2], s[3], s[4], s[5], a] + \
        eta * (r + gma * np.max(Q[s_next[0], s_next[1], s_next[2], s_next[3], s_next[4], s_next[5], :]))


def learn_sarsa(Q, state, state_next, action, action1, eta, gma, r):
    Q[state[0], state[1], state[2], state[3], state[4], state[5], action] = \
        Q[state[0], state[1], state[2], state[3], state[4], state[5], action] + \
        eta * (r + gma * Q[state_next[0], state_next[1], state_next[2], state_next[3], state_next[4], state_next[5], action1] -
               Q[state[0], state[1], state[2], state[3], state[4], state[5], action])

