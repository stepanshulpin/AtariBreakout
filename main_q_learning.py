from game import Game
from q_learning import QLearning

q = QLearning(.628, .9)

game1 = Game()

print("learn")
q.learn(game1, 20000)
print("play")
game2 = Game()
q.play(game2, 1000)