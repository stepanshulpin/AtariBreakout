import gym

from utils import get_state, convert_to_direction


class Game(object):

    def __init__(self):
        self.env = gym.make('Breakout-v0')
        self.actions = range(self.env.action_space.n)
        self.rev_list = list()

        self.lives_prev = 5
        self.rAll = 0

        self.info = None
        self.info_next = None
        self.d = None
        self.d_next = None
        self.r = None
        self.r_next = None

    def get_action_n(self):
        return self.env.action_space.n

    def add_reward(self):
        self.rev_list.append(self.rAll)
        self.rAll = 0

    def add_r(self):
        self.rAll += max(self.r, self.r_next)

    def print_statistic(self, step):
        print("Reward Sum on " + str(step) + " episodes " + str(sum(self.rev_list) / step))
        print("Reward Max on " + str(step) + " episodes " + str(max(self.rev_list)))

    def is_fail(self):
        lives = self.info.get('ale.lives')
        lives_next = self.info_next.get('ale.lives')
        if self.lives_prev > lives or self.lives_prev > lives_next:
            if lives_next == 0:
                self.lives_prev = 5
            else:
                self.lives_prev = lives_next
            return True
        else:
            return False

    def not_end(self):
        return not self.d and not self.d_next

    def is_break(self):
        return self.r != 0 or self.r_next != 0

    def get_next_state(self, a):
        s_next, r, d, info = self.env.step(a)

        s_next_none, r_next, d_next, info_next = self.env.step(0)
        s_next = get_state(s_next)
        s_next_none = get_state(s_next_none)
        s_next_none = convert_to_direction(s_next, s_next_none)
        s_next = s_next + s_next_none

        self.env.render()

        self.info = info
        self.info_next = info_next

        self.d = d
        self.d_next = d_next

        self.r = r
        self.r_next = r_next

        return s_next

    def reset_game(self):
        s = self.env.reset()
        s_none, _, _, _ = self.env.step(0)
        s = get_state(s)
        s_none = get_state(s_none)
        s_none = convert_to_direction(s, s_none)
        s = s + s_none
        self.env.render()
        return s
