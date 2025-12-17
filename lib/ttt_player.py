# Player model for TicTacToe board
from . import ml
from . import replay_buffer
import copy
import math
import json
import random
from utils import compress, decompress

from listinvert import value, Matrix, multiply_matrix, Mod3l, Block, Data, MatMul, SSE, Reshape, Sigmoid, Add, BCE

START_VALUES = [
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
]
# For some reason this does not work
# START_VALUES = [ [None]*6 for _ in range(6)]


def DData(mod3l, rows, cols, values):
    res = Data(mod3l, rows, cols)
    mod3l.set_data(res, values)
    return res


class TTTRandom:
    def get_next_step_values(self, boards):
        values = copy.deepcopy(START_VALUES)
        for board, x, y in boards:
            values[x][y] = self.get_next_step_value(b)
        return values

    def get_next_step_value(self, board):
        return random.random()


# Simple player based on board position value
class TTTPlayer:
    def __init__(self, file_to_load_from=None, enable_cpp=False):
      self.replay_buffer = replay_buffer.ReplayBuffer(max_size=10000)

      model_json = None
      if file_to_load_from:
          model_json = self.parse_model_file(file_to_load_from)
          self.enable_cpp = model_json.get("cpp", False)
      else:
          self.enable_cpp = enable_cpp

      if self.enable_cpp:
           self.m = Mod3l()
           self.x = Data(self.m, 6, 6)

           self.w1 = Data(self.m, 36, 64)
           self.b1 = Data(self.m, 1, 64)

           self.w2 = Data(self.m, 64, 32)
           self.b2 = Data(self.m, 1, 32)

           self.w3 = Data(self.m, 32, 1)
           self.b3 = Data(self.m, 1, 1)

           if not model_json:
             self.m.set_data(self.x, ml.random_matrix(6, 6)) 
             self.m.set_data(self.w1, ml.random_matrix(36, 64))
             self.m.set_data(self.b1, ml.random_matrix(1, 64))
             self.m.set_data(self.w2, ml.random_matrix(64, 32))
             self.m.set_data(self.b2, ml.random_matrix(1, 32))
             self.m.set_data(self.w3, ml.random_matrix(32, 1))
             self.m.set_data(self.b3, ml.random_matrix(1, 1))


           self.z0 = Reshape(self.x, 1, 36)
           self.z1 = Sigmoid(Add(MatMul(self.z0, self.w1), self.b1))
           self.z2 = Sigmoid(Add(MatMul(self.z1, self.w2), self.b2))
           # this 3 lines instead of 1 is only for the test showcase
           #z3 = Sigmoid(Add(MatMul(z2, w3), b3))
           self.zm = MatMul(self.z2, self.w3)
           self.za = Add(self.zm, self.b3)
           self.z3 = Sigmoid(self.za)

           self.y = DData(self.m, 1, 1, ml.random_matrix(1, 1))
           self.loss = BCE(self.z3, self.y)

      else:
        self.x = ml.BB(ml.random_matrix(6, 6))

        self.w1 = ml.BB(ml.random_matrix(36, 64))
        self.b1 = ml.BB(ml.random_matrix(1, 64))

        self.w2 = ml.BB(ml.random_matrix(64, 32))
        self.b2 = ml.BB(ml.random_matrix(1, 32))

        self.w3 = ml.BB(ml.random_matrix(32, 1))
        self.b3 = ml.BB(ml.random_matrix(1, 1))

        self.z0 = ml.BBReshape(self.x, 1, 36)
        self.z1 = (self.z0 @ self.w1 + self.b1).sigmoid()
        self.z2 = (self.z1 @ self.w2 + self.b2).sigmoid()
        self.z3 = (self.z2 @ self.w3 + self.b3).sigmoid()

        # Predicts who is the winner (1 for crosses, -1 for zeroes)
        self.prediction = self.z3

        self.y = ml.BB(ml.random_matrix(1, 1))
        # self.loss = self.prediction.mse(self.y)
        self.loss = self.prediction.bce(self.y)

      if model_json:
        self.load_from_json(model_json)

    def parse_model_file(self, file_name):
        with open(file_name, "r") as file:
            return json.loads(file.read())

    def load_from_file(self, file_name):
        self.load_from_json(self.parse_model_file(file_name))

    def load_from_json(self, model_json):
            if isinstance(model_json, list):
                # Old format
                self.loss.from_json(model_json)
            else:
                if model_json.get("cpp", False):
                    self.enable_cpp = True
                    data = model_json["data"]
                    self.m.set_data(self.w1, data["w1"])
                    self.m.set_data(self.w2, data["w2"])
                    self.m.set_data(self.w3, data["w3"])
                    self.m.set_data(self.b1, data["b1"])
                    self.m.set_data(self.b2, data["b2"])
                    self.m.set_data(self.b3, data["b3"])
                else:
                    self.loss.from_json(model_json["data"])

                if "replay_buffer" in model_json:
                    self.replay_buffer.from_json(model_json["replay_buffer"])
                else:
                    self.replay_buffer.from_json(decompress(model_json["replay_buffer_zip"]))



    def calc_grads(self):
        if self.enable_cpp:
           pass  # it is now calculate under the hood whenver needed
        else:
           self.loss.dif()
            

    def save_to_file(self, file_name):
        def rounded(mtx):
            return [[round(x, 5) for x in row] for row in mtx]

        with open(file_name, "w") as file:
            if self.enable_cpp:
               data_json = {
                  "w1": rounded(value(self.w1.fval())),
                  "w2": rounded(value(self.w2.fval())),
                  "w3": rounded(value(self.w3.fval())),
                  "b1": rounded(value(self.b1.fval())),
                  "b2": rounded(value(self.b2.fval())),
                  "b3": rounded(value(self.b3.fval())),
               }
            else:
               data_json = self.loss.to_json()
            model_json = {
                "cpp": self.enable_cpp,
                "data": data_json,
                "replay_buffer_zip": compress(self.replay_buffer.to_json()),
            }
            model_dump = json.dumps(model_json)
            file.write(model_dump)

    # For a set of next step boards and coords of next step
    # calculates value and stores it in the coords of next step.
    def get_next_step_values(self, boards):
        values = copy.deepcopy(START_VALUES)
        for bxy in boards:
            b, x, y = bxy
            if self.enable_cpp:
                self.m.set_data(self.x, b)
                value = value(self.z3.fval())
            else:
                self.x.set(b)
                value = self.prediction.val()
            values[x][y] = value[0][0]
        return values


    def get_next_step_values(self, boards):
        values = copy.deepcopy(START_VALUES)
        for board, x, y in boards:
            values[x][y] = self.get_next_step_value(board)
        return values

    def get_next_step_value(self, board):
            if self.enable_cpp:
                self.m.set_data(self.x, board)
                step_value = value(self.z3.fval())
            else:
                self.x.set(board)
                step_value = self.prediction.val()
            return step_value[0][0]

    def get_loss_value(self):
        return value(self.loss.fval())[0][0]

    def apply_gradient(self, alpha = 0.01):

      # TODO: interesting! explore more
      # norm = lambda matrix: math.sqrt(sum(sum(x**2 for x in row) for row in matrix))
      # w1norm = norm(self.w1.val())
      # w1dnorm = norm(self.w1.dval())
      # alpha = 0.01
      # if w1dnorm / w1norm < 0.01:
      #    alpha = 0.1
      # if w1dnorm / w1norm > 1000:
      #    alpha = 0.001
      # alpha = 0.0001
      if not self.enable_cpp: 
        self.w1.appl(alpha)
        self.b1.appl(alpha)
        self.w2.appl(alpha)
        self.b2.appl(alpha)
        self.w3.appl(alpha)
        self.b3.appl(alpha)
      else:
        self.w1.apply_bval(alpha)
        self.b1.apply_bval(alpha)
        self.w2.apply_bval(alpha)
        self.b2.apply_bval(alpha)
        self.w3.apply_bval(alpha)
        self.b3.apply_bval(alpha)

