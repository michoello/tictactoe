# Classifier model for TicTacToe board
from . import ml


# Simple classifier of board position
class TTTClass:
  def __init__(self):

    self.x = ml.BB(ml.random_matrix(6,6))

    self.w1 = ml.BB(ml.random_matrix(36, 16))
    self.b1 = ml.BB(ml.random_matrix(1, 16))

    self.w2 = ml.BB(ml.random_matrix(16, 1))
    self.b2 = ml.BB(ml.random_matrix(1, 1))

    #self.w3 = ml.BB(ml.random_matrix(4, 1))
    #self.b3 = ml.BB(ml.random_matrix(1, 1))


    self.z0 = ml.BBReshape(self.x, 1, 36)
    self.z1 = (self.z0 @ self.w1 + self.b1).sigmoid()
    self.z2 = (self.z1 @ self.w2 + self.b2).sigmoid()
    #self.z3 = (self.z2 @ self.w3 + self.b3).sigmoid()


    # Predicts who is the winner (1 for crosses, -1 for zeroes)
    self.prediction = self.z2

    self.y = ml.BB(ml.random_matrix(1, 1))
    self.loss = self.prediction.mse(self.y)


  def load_from_file(self, file_name):
     with open(file_name, "r") as file:
       model_dump = file.read()
       self.loss.load(model_dump)

