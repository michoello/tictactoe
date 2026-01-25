# Player model for TicTacToe board
# Version 2:
#   - Convolutions with ReLU
#   - Value through Tanh
#   - Policy output

from . import ml
from . import replay_buffer
import copy
import math
import json
import random
from utils import compress, decompress

from listinvert import (
    Matrix,
    multiply_matrix,
    Mod3l,
    Block,
    Data,
    MatMul,
    SSE,
    Abs,
    Add,
    BCE,
    Sigmoid,
    Reshape,
    value,
    Convo,
    ReLU,
    SoftMax,
    SoftMaxCrossEntropy,
    Tanh,
)

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

# More complex player model, producing value and policy.
# Works both sides as it converts board differently depending on whose move is next
class TTTPlayerV2:
    def __init__(self, spec_file=None):
        self.impl = TTTPlayerImpl(spec_file)

    def get_next_step_value(self, player, board):
        return self.impl.get_next_step_value(board)

    def calc_grads(self):
        pass

    def apply_gradient(self, alpha = 0.01):
        self.impl.apply_gradient(alpha)

    # Board is 6*6 matrix of -1 for Os, 1 for Xs, 0 for empty cells
    # Value is 1*1 matrix with the board reward, i.e. [-1 to 1]
    def set_board_and_value(self, player, board, _value):
        self.impl.m.set_data(impl.x, board)
        self.impl.m.set_data(impl.y, _value)

    def save_to_file(self, file_name):
        self.impl.save_to_file(file_name)

    def replay_buffer(self):
        return self.impl.replay_buffer

    def get_loss_value(self, player):
        return self.impl.get_loss_value()


# Single side player based on board position value
# Can play either for X xor for O, but not both
# Became private implementation detail, not to be used directly
class TTTPlayerImpl:
    def __init__(self, file_to_load_from=None):
       self.replay_buffer = replay_buffer.ReplayBuffer(max_size=10000)

       self.m = Mod3l()

       self.dinput = Data(self.m, 3, 3)
       self.dkernel1 = Data(self.m, 2, 2)
       self.dc1 = Convo(self.dinput, self.dkernel1)
       self.rl1 = ReLU(self.dc1)

       self.dkernel2 = Data(self.m, 2, 2)
       self.dc2 = Convo(self.dinput, self.dkernel2)
       self.rl2 = ReLU(self.dc2)

       self.rl = Add(self.rl1, self.rl2)

       self.dw = Data(self.m, 3, 3)
       self.dlogits = MatMul(self.rl, self.dw)
       self.dsoftmax = SoftMax(self.dlogits)

       # Policy labels
       self.dlabels = Data(self.m, 3, 3)
       self.policy_loss = SoftMaxCrossEntropy(self.dlogits, self.dsoftmax, self.dlabels)

        
       self.dw2 = Data(self.m, 3, 1)
       self.dvalue = Tanh(MatMul(self.rl, self.dw2))
       # Value label
       self.dlabel = Data(self.m, 1, 1)
       self.dvalue_loss = SSE(self.dvalue, self.dlabel)

       """
       self.x = Data(self.m, 6, 6)

       self.w1 = Data(self.m, 36, 64)
       self.b1 = Data(self.m, 1, 64)

       self.w2 = Data(self.m, 64, 32)
       self.b2 = Data(self.m, 1, 32)

       self.w3 = Data(self.m, 32, 1)
       self.b3 = Data(self.m, 1, 1)



       self.z0 = Reshape(self.x, 1, 36)
       self.z1 = Sigmoid(Add(MatMul(self.z0, self.w1), self.b1))
       self.z2 = Sigmoid(Add(MatMul(self.z1, self.w2), self.b2))

       self.zm = MatMul(self.z2, self.w3)
       self.za = Add(self.zm, self.b3)
       self.z3 = Sigmoid(self.za)

       self.y = DData(self.m, 1, 1, ml.random_matrix(1, 1))
       self.loss = BCE(self.z3, self.y)

       if file_to_load_from:
           self.load_from_file(file_to_load_from)
       else:
           self.m.set_data(self.x, ml.random_matrix(6, 6)) 
           self.m.set_data(self.w1, ml.random_matrix(36, 64))
           self.m.set_data(self.b1, ml.random_matrix(1, 64))
           self.m.set_data(self.w2, ml.random_matrix(64, 32))
           self.m.set_data(self.b2, ml.random_matrix(1, 32))
           self.m.set_data(self.w3, ml.random_matrix(32, 1))
           self.m.set_data(self.b3, ml.random_matrix(1, 1))
       """

    def parse_model_file(self, file_name):
        with open(file_name, "r") as file:
            return json.loads(file.read())

    def load_from_file(self, file_name):
        self.load_from_json(self.parse_model_file(file_name))

    def load_from_json(self, model_json):
        data = model_json["data"]
        self.m.set_data(self.w1, data["w1"])
        self.m.set_data(self.w2, data["w2"])
        self.m.set_data(self.w3, data["w3"])
        self.m.set_data(self.b1, data["b1"])
        self.m.set_data(self.b2, data["b2"])
        self.m.set_data(self.b3, data["b3"])

        if "replay_buffer" in model_json:
            self.replay_buffer.from_json(model_json["replay_buffer"])
        else:
            self.replay_buffer.from_json(decompress(model_json["replay_buffer_zip"]))


    def calc_grads(self):
        pass
            

    def save_to_file(self, file_name):
        def rounded(mtx):
            return [[round(x, 5) for x in row] for row in mtx]

        with open(file_name, "w") as file:
            data_json = {
                  "w1": rounded(value(self.w1.fval())),
                  "w2": rounded(value(self.w2.fval())),
                  "w3": rounded(value(self.w3.fval())),
                  "b1": rounded(value(self.b1.fval())),
                  "b2": rounded(value(self.b2.fval())),
                  "b3": rounded(value(self.b3.fval())),
            }
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
        for board, x, y in boards:
            values[x][y] = self.get_next_step_value(board)
        return values

    def get_next_step_value(self, board):
        self.m.set_data(self.x, board)
        step_value = value(self.z3.fval())
        return step_value[0][0] 

    def get_loss_value(self):
        return value(self.loss.fval())[0][0]

    def apply_gradient(self, alpha = 0.01):
        self.w1.appl(alpha)
        self.b1.appl(alpha)
        self.w2.appl(alpha)
        self.b2.appl(alpha)
        self.w3.appl(alpha)
        self.b3.appl(alpha)
