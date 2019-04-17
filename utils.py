import matplotlib.pyplot as plt
import numpy as np


def show(s):
    fig, axes = plt.subplots()
    axes.imshow(s)
    plt.show()


def to_grayscale(img):
    n = img.shape[0]
    m = img.shape[1]
    res_img = np.zeros([n, m], dtype=np.bool_)
    for i in range(n):
        for j in range(m):
            pixel = img[i, j, :]
            if any(p > 0 for p in pixel):
                res_img[i, j] = 1
    return res_img


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
    l = list(img[0, ::4])
    #len = 0
    pos = 0
    pos = l[pos:].index(1)
    #len = l[pos:].index(0)
    #assert len == 2, 'len is odd'
    return [pos]


def ball_pos(img):
    n = img.shape[0]
    m = img.shape[1]
    pos = [12, 18]
    for i in range(n):
        for j in range(m):
            if img[i, j] == 1:
                pos = [int(i/2), int(j/4)]
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
