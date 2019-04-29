import numpy as np

import utils


class QLearning(object):
    def __init__(self, eta, gma):
        self.eta = eta
        self.gma = gma
        self.Q = None
        self.game = None
        self.last_states = list()
        self.last_next_states = list()
        self.last_actions = list()

    def learn(self, game, epis):
        self.Q = np.zeros([25, 73, 72, 3, 3, 3, game.get_action_n()])
        self.game = game
        for i in range(epis):
            s = game.reset_game()

            restart = False
            game.d = False
            game.d_next = False
            while game.not_end():

                if restart:
                    a = 1
                    restart = False
                else:
                    a = self.get_opt_action(s, i)

                s_next = game.get_next_state(a)

                if s == s_next:
                    restart = True

                self.last_states.append(s)
                self.last_next_states.append(s_next)
                self.last_actions.append(a)

                if game.is_break():
                    self.assign_reward(1, 15)

                if game.is_fail():
                    self.assign_reward(-2, 8)

                game.add_r()
                s = s_next

            game.add_reward()
            if i > 0 and i % 100 == 0:
                game.print_statistic(i)
                print("Q Sum on " + str(i) + " episodes " + str(np.sum(self.Q)))

        game.print_statistic(epis)
        np.save("Q_Table.npy", self.Q)

    def play(self, game, epis):

        self.Q = np.load("Q_Table.npy")

        game.d = False
        game.d_next = False
        for i in range(epis):
            s = game.reset_game()
            while game.not_end():
                a = self.get_max_action(s)

                s_next = game.get_next_state(a)
                s = s_next

            if i > 0 and i % 100 == 0:
                game.print_statistic(i)

        game.print_statistic(epis)

    def get_opt_action(self, s, i):
        return np.argmax(self.Q[s[0], s[1], s[2], s[3], s[4], s[5], :]
                         + np.random.randn(1, self.game.get_action_n()) * (0.1 / (i + 1)))

    def get_max_action(self, s):
        return np.argmax(self.Q[s[0], s[1], s[2], s[3], s[4], s[5], :])

    def assign_reward(self, r, n):
        k = 0
        while k < n and len(self.last_states) > 0:
            state = self.last_states.pop()
            state_next = self.last_next_states.pop()
            action = self.last_actions.pop()
            utils.learn(self.Q, state, state_next, action, self.eta, self.gma, r)

        self.last_states.clear()
        self.last_next_states.clear()
        self.last_actions.clear()