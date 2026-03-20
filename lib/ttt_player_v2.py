# Player model for TicTacToe board
# Version 2:
#   - Convolutions with ReLU
#   - Value through Tanh
#   - Policy output

from . import ml
from .game import GameState
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
    GradClipper,
    SSE,
    Abs,
    Add,
    BCE,
    Sigmoid,
    Reshape,
    Explode,
    MulEl2,
    value,
    Convo,
    Convo2,
    ReLU,
    SoftMax,
    SoftMaxCrossEntropy,
    Tanh,
)

from typing import Optional
START_VALUES: list[list[Optional[float]]] = [
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
    [None, None, None, None, None, None],
]
# For some reason this does not work
# START_VALUES_COMMENTED = [ [None]*6 for _ in range(6)]

def DData(mod3l: Mod3l, rows: int, cols: int, values: list[list[float]]) -> Data:
    res = Data(mod3l, rows, cols)
    mod3l.set_data(res, values)
    return res

CONVO_CHANNELS = 32

# More complex player model, producing value and policy.
# Works both sides as it converts board differently depending on whose move is next
class TTTPlayerV2:
    def __init__(self, spec_file: Optional[str] = None) -> None:
       self._replay_buffer = replay_buffer.ReplayBuffer(max_size=10000)

       self.m = Mod3l()
       self.dinput = Data(self.m, 6, 6)
       self.dplayer = Data(self.m, 1, 1)

       Nonlinearity = Sigmoid  # ReLU does not work though recommended by BOOKS

       # Common trunk
       self.fold = Data(self.m, CONVO_CHANNELS, 1)

       self.kernels1 = Data(self.m, 9, CONVO_CHANNELS)
       self.kernels2 = Data(self.m, 9, CONVO_CHANNELS)

       scaled_input = MulEl2(self.dinput, self.dplayer)
       rl = MatMul(Explode(scaled_input, 3, 3), self.kernels1)
       #rl = MatMul(Explode(GradClipper(self.dinput, 1.0), 3, 3), GradClipper(self.kernels1, 1.0))
       rl = Reshape(rl, 6, 6)
       rl = Nonlinearity(rl)

       rl = MatMul(Explode(rl, 3, 3), self.kernels2) # dims:36,CONVO_CHANNELS
       #rl = MatMul(Explode(GradClipper(rl, 1.0), 3, 3), GradClipper(self.kernels2, 1.0)) # dims:36,CONVO_CHANNELS
       rl = Nonlinearity(rl) 

       rl = MatMul(rl, self.fold)  # dims:36,1

       self.rl_flat = Reshape(rl, 1, 36)

       # Policy
       self.w_policy = Data(self.m, 36, 36)
       self.b_policy = Data(self.m, 1, 36)
       self.policy_logits = Add(MatMul(self.rl_flat, self.w_policy), self.b_policy)
       self.policy = SoftMax(self.policy_logits)

       self.policy_labels = Data(self.m, 1, 36)
       self.policy_loss = SoftMaxCrossEntropy(self.policy_logits, self.policy, self.policy_labels)
       self.rl1 = self.policy # !!!

       # Value
       self.w_value1 = Data(self.m, 36, 36)
       self.b_value1 = Data(self.m, 1, 36)
       v = Nonlinearity(Add(MatMul(self.rl_flat, self.w_value1), self.b_value1))
       
       self.w_value = Data(self.m, 36, 1)
       self.b_value = Data(self.m, 1, 1)
       self.value = Tanh(Add(MatMul(v, self.w_value), self.b_value))

       self.value_label = Data(self.m, 1, 1) 
       self.value_loss = SSE(self.value, self.value_label)

       if spec_file:
           self.load_from_file(spec_file)
       else:
           # Convolutions should be initialized very carefully to prevent gradient crazyness
           # TODO: add batch normalization and gradient clipping
           #self.m.set_data(self.kernels1, ml.random_matrix(9, CONVO_CHANNELS, 0.0005))
           #self.m.set_data(self.kernels2, ml.random_matrix(9, CONVO_CHANNELS, 0.0005))
           self.m.set_data(self.kernels1, ml.random_matrix(9, CONVO_CHANNELS, 1.0))
           self.m.set_data(self.kernels2, ml.random_matrix(9, CONVO_CHANNELS, 1.0))

           self.m.set_data(self.w_policy, ml.random_matrix(36, 36))
           self.m.set_data(self.b_policy, ml.random_matrix(1, 36))
           self.m.set_data(self.w_value1, ml.random_matrix(36, 36))
           self.m.set_data(self.b_value1, ml.random_matrix(1, 36))
           self.m.set_data(self.w_value, ml.random_matrix(36, 1))
           self.m.set_data(self.b_value, ml.random_matrix(1, 1))

           # TODO: simply moving this line up to the kernels1,2 initialization breaks
           # the game debugging test entirely (just because of different values).
           # A better test and initialization is needed to make sure the training is more or less
           # stable
           #self.m.set_data(self.fold, ml.random_matrix(CONVO_CHANNELS, 1, 0.1))
           self.m.set_data(self.fold, ml.random_matrix(CONVO_CHANNELS, 1, 1.0))


    def get_next_step_values(self, player: int, boards: list[tuple[Matrix, int, int]]) -> list[list[Optional[float]]]:
        values = copy.deepcopy(START_VALUES)
        for board, x, y in boards:
            values[x][y] = self.get_next_step_value(player, board)
        return values


    def get_next_step_value(self, player: int, board: Matrix) -> float:
        self.m.set_data(self.dplayer, [[player]])
        self.m.set_data(self.dinput, board)
        #step_value = value(self.value_label.fval())
        step_value = value(self.value.fval())
        return player * step_value[0][0]


    def calc_grads(self) -> None:
        pass


    def apply_gradient(self, alpha: float = 0.01) -> None:
        self.kernels1.apply_bval(alpha)
        self.kernels2.apply_bval(alpha)
        self.fold.apply_bval(alpha)

        self.w_policy.apply_bval(alpha)
        self.b_policy.apply_bval(alpha)
        self.w_value1.apply_bval(alpha)
        self.b_value1.apply_bval(alpha)
        self.w_value.apply_bval(alpha)
        self.b_value.apply_bval(alpha)


    # Board is 6*6 matrix of -1 for Os, 1 for Xs, 0 for empty cells
    # Value is 1*1 matrix with the board reward, i.e. [-1 to 1]
    def set_board_and_value(self, player: int, state: GameState, policy: Optional[list[list[float]]] = None) -> None:
        board = state.board.cells
        _value = state.reward
        self.m.set_data(self.dplayer, [[player]])
        self.m.set_data(self.dinput, board)

        _value = _value or [[0.5]]
        _value = [[_value[0][0] * player]]
        self.m.set_data(self.value_label, _value)

        policy = policy or [ [1/36 for _ in range(36)]]
        self.m.set_data(self.policy_labels, policy)


    def parse_model_file(self, file_name: str) -> dict:
        with open(file_name, "r") as file:
            return json.loads(file.read())

    def load_from_file(self, file_name: str) -> None:
        self.load_from_json(self.parse_model_file(file_name))

    def load_from_json(self, model_json: dict) -> None:
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
            self._replay_buffer.from_json(model_json["replay_buffer"])
        else:
            self._replay_buffer.from_json(decompress(model_json["replay_buffer_zip"]))


    def replay_buffer(self) -> replay_buffer.ReplayBuffer:
        return self._replay_buffer

    def save_to_file(self, file_name: str) -> None:
        def rounded(mtx: list[list[float]]) -> list[list[float]]:
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
               "kernel1": rounded(value(self.kernels1.fval())),
               "kernel2": rounded(value(self.kernels2.fval())),
               "fold":    rounded(value(self.fold.fval())),
            }

            model_json = {
                "data": data_json,
                "replay_buffer_zip": compress(self._replay_buffer.to_json()),
            }
            model_dump = json.dumps(model_json)
            file.write(model_dump)


    # Returns tuple of two losses: value and policy
    def get_loss_value(self) -> tuple[float, float]:
        return self.value_loss.fval().get(0, 0), self.policy_loss.fval().get(0, 0)


    # Somehow this function never triggered.
    # TODO: remove it?
    def find_nan_grads(self) -> bool:
       def contains_nan(block: Block) -> bool:
           return any(math.isnan(x) for row in value(block.bval()) for x in row)

       blocks = {
           "w_policy": self.w_policy,
           "b_policy": self.b_policy,
           "w_value1": self.w_value1,
           "b_value1": self.b_value1,
           "w_value": self.w_value,
           "b_value": self.b_value,
           "kernel1": self.kernels1,
           "kernel2": self.kernels2,
           "fold":    self.fold,
       }
       for name, block in blocks.items():
           if contains_nan(block):
               print(f"Block {name} contains nan!")
               return True
       return False

