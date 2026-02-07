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
    Explode,
    value,
    Convo,
    Convo2,
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

CONVO_CHANNELS = 32

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
    def set_board_and_value(self, player, board, _value=None, policy=None):
        self.impl.m.set_data(self.impl.dinput, board)
        if _value:
           self.impl.m.set_data(self.impl.value_label, _value)
        if policy:
           self.impl.m.set_data(self.impl.policy_labels, policy)


    def save_to_file(self, file_name):
        self.impl.save_to_file(file_name)

    def replay_buffer(self):
        return self.impl.replay_buffer

    # Returns tuple of two losses: value and policy
    def get_loss_value(self):
        return self.impl.get_loss_value()


# Single side player based on board position value
# Can play either for X xor for O, but not both
# Became private implementation detail, not to be used directly
class TTTPlayerImpl:
    def __init__(self, file_to_load_from=None):
       self.replay_buffer = replay_buffer.ReplayBuffer(max_size=10000)

       self.m = Mod3l()
       self.dinput = Data(self.m, 6, 6)

       Nonlinearity = Sigmoid  # ReLU does not work though recommended by BOOKS

       def Convo3(input, kernel):
          return Reshape(MatMul(Explode(input, 3, 3), Reshape(kernel,  9, 1)), 6, 6)

       # Common trunk
       self.fold = Data(self.m, CONVO_CHANNELS, 1)

       self.kernels1 = Data(self.m, 9, CONVO_CHANNELS)
       self.kernels2 = Data(self.m, 9, CONVO_CHANNELS)

       rl = Reshape(MatMul(Explode(self.dinput, 3, 3), self.kernels1), 6, 6)
       rl = Nonlinearity(rl)

       rl = MatMul(Explode(rl, 3, 3), self.kernels2)
       rl = Nonlinearity(rl) 

       rl = MatMul(rl, self.fold)
       #print("AAAAAAAAAAAAAAAAAAAAAAAAA", rl.rows(), rl.cols())

       self.rl = rl #Reshape(rl, 6, 6)
       self.rl_flat = Reshape(self.rl, 1, 36)

       # Policy
       self.w_policy = Data(self.m, 36, 36)
       self.b_policy = Data(self.m, 1, 36)
       self.policy_logits = Add(MatMul(self.rl_flat, self.w_policy), self.b_policy)
       self.policy = SoftMax(self.policy_logits)

       self.policy_labels = Data(self.m, 1, 36)
       self.policy_loss = SoftMaxCrossEntropy(self.policy_logits, self.policy, self.policy_labels)

       # Value
       self.w_value1 = Data(self.m, 36, 36)
       self.b_value1 = Data(self.m, 1, 36)
       v = Nonlinearity(Add(MatMul(self.rl_flat, self.w_value1), self.b_value1))
       
       self.w_value = Data(self.m, 36, 1)
       self.b_value = Data(self.m, 1, 1)
       self.value = Tanh(Add(MatMul(v, self.w_value), self.b_value))

       self.value_label = Data(self.m, 1, 1) 
       self.value_loss = SSE(self.value, self.value_label)

       if file_to_load_from:
           self.load_from_file(file_to_load_from)
       else:
           self.m.set_data(self.kernels1, ml.random_matrix(9, CONVO_CHANNELS))
           self.m.set_data(self.kernels2, ml.random_matrix(9, CONVO_CHANNELS))

           self.m.set_data(self.fold, [[1] for _ in range(CONVO_CHANNELS)])

           self.m.set_data(self.w_policy, ml.random_matrix(36, 36))
           self.m.set_data(self.b_policy, ml.random_matrix(1, 36))
           self.m.set_data(self.w_value1, ml.random_matrix(36, 36))
           self.m.set_data(self.b_value1, ml.random_matrix(1, 36))
           self.m.set_data(self.w_value, ml.random_matrix(36, 1))
           self.m.set_data(self.b_value, ml.random_matrix(1, 1))


    def parse_model_file(self, file_name):
        with open(file_name, "r") as file:
            return json.loads(file.read())

    def load_from_file(self, file_name):
        self.load_from_json(self.parse_model_file(file_name))

    def load_from_json(self, model_json):
        data = model_json["data"]

        self.m.set_data(self.kernels1, data[f"kernel1"])
        self.m.set_data(self.kernels2, data[f"kernel2"])
        self.m.set_data(self.fold, data["fold"])

        self.m.set_data(self.w_policy, data["w_policy"])
        self.m.set_data(self.b_policy, data["b_policy"])

        self.m.set_data(self.w_value1, data["w_value1"])
        self.m.set_data(self.b_value1, data["b_value1"])
        self.m.set_data(self.w_value, data["w_value"])
        self.m.set_data(self.b_value, data["b_value"])

        if "replay_buffer" in model_json:
            self.replay_buffer.from_json(model_json["replay_buffer"])
        else:
            self.replay_buffer.from_json(decompress(model_json["replay_buffer_zip"]))


    def calc_grads(self):
        pass
            

    def save_to_file(self, file_name):
        def rounded(mtx):
            #return [[round(x, 5) for x in row] for row in mtx]
            return [[x for x in row] for row in mtx]

        with open(file_name, "w") as file:
            data_json = {
               "w_policy": rounded(value(self.w_policy.fval())),
               "b_policy": rounded(value(self.b_policy.fval())),
               "w_value1": rounded(value(self.w_value1.fval())),
               "b_value1": rounded(value(self.b_value1.fval())),
               "w_value": rounded(value(self.w_value.fval())),
               "b_value": rounded(value(self.b_value.fval())),
               "fold":    rounded(value(self.fold.fval())),
            }
            data_json[f"kernel1"] = rounded(value(self.kernels1.fval()))
            data_json[f"kernel2"] = rounded(value(self.kernels2.fval()))

            model_json = {
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
        self.m.set_data(self.dinput, board)
        step_value = value(self.value_label.fval())
        return step_value[0][0] 

    def get_loss_value(self):
        return self.value_loss.fval().get(0, 0), self.policy_loss.fval().get(0, 0)

    def apply_gradient(self, alpha = 0.01):
        self.kernels1.apply_bval(alpha)
        self.kernels2.apply_bval(alpha)

        self.w_policy.apply_bval(alpha)
        self.b_policy.apply_bval(alpha)
        self.w_value1.apply_bval(alpha)
        self.b_value1.apply_bval(alpha)
        self.w_value.apply_bval(alpha)
        self.b_value.apply_bval(alpha)

