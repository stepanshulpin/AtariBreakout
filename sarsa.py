import numpy as np

import utils


class Sarsa(object):
    def __init__(self, eta, gma):
        self.eta = eta
        self.gma = gma
        self.Q = None
        self.game = None
        self.last_states = list()
        self.last_next_states = list()
        self.last_actions = list()
        self.last_actions_1 = list()

    def learn(self, game, epis):
        self.Q = np.zeros([25, 73, 72, 3, 3, 3, game.get_action_n()])
        self.game = game
        for i in range(epis):
            s = game.reset_game()
            a = self.get_opt_action(s, i)
            restart = False
            game.d = False
            game.d_next = False
            while game.not_end():
                s_next = game.get_next_state(a)
                if restart:
                    a1 = 1
                    restart = False
                else:
                    a1 = self.get_opt_action(s_next, i)

                if s == s_next:
                    restart = True

                self.last_states.append(s)
                self.last_next_states.append(s_next)
                self.last_actions.append(a)
                self.last_actions_1.append(a1)

                if game.is_break():
                    self.assign_reward(1, 15)

                if game.is_fail():
                    self.assign_reward(-2, 8)

                game.add_r()
                s = s_next
                a = a1
            game.add_reward()
            if i > 0 and i % 100 == 0:
                game.print_statistic(i)
                print("Q Sum on " + str(i) + " episodes " + str(np.sum(self.Q)))

        game.print_statistic(epis)
        np.save("Q_Table_SARSA.npy", self.Q)

    def play(self, game, epis):

        self.Q = np.load("Q_Table_SARSA.npy")

        for i in range(epis):
            s = game.reset_game()

            game.d = False
            game.d_next = False
            while game.not_end():
                a = self.get_max_action(s)

                s_next = game.get_next_state(a)

                game.add_r()
                s = s_next
            game.add_reward()
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
            action1 = self.last_actions_1.pop()
            utils.learn_sarsa(self.Q, state, state_next, action, action1, self.eta, self.gma, r)

        self.last_states.clear()
        self.last_next_states.clear()
        self.last_actions.clear()
        self.last_actions_1.clear()