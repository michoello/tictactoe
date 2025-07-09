# Player model for TicTacToe board
from . import ml
from . import replay_buffer
import copy
import math

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


# Simple player based on board position value
class TTTPlayer:
    def __init__(self, file_to_load_from=None):

        self.replay_buffer = replay_buffer.ReplayBuffer(max_size=10000)

        self.x = ml.BB(ml.random_matrix(6, 6))

        self.w1 = ml.BB(ml.random_matrix(36, 64))
        self.b1 = ml.BB(ml.random_matrix(1, 64))

        self.w2 = ml.BB(ml.random_matrix(64, 1))
        self.b2 = ml.BB(ml.random_matrix(1, 1))

        # self.w3 = ml.BB(ml.random_matrix(32, 1))
        # self.b3 = ml.BB(ml.random_matrix(1, 1))

        self.z0 = ml.BBReshape(self.x, 1, 36)
        self.z1 = (self.z0 @ self.w1 + self.b1).sigmoid()
        self.z2 = (self.z1 @ self.w2 + self.b2).sigmoid()
        # self.z3 = (self.z2 @ self.w3 + self.b3).sigmoid()

        # Predicts who is the winner (1 for crosses, -1 for zeroes)
        self.prediction = self.z2

        self.y = ml.BB(ml.random_matrix(1, 1))
        # self.loss = self.prediction.mse(self.y)
        self.loss = self.prediction.bce(self.y)

        if file_to_load_from is not None:
            self.load_from_file(file_to_load_from)

    def load_from_file(self, file_name):
        with open(file_name, "r") as file:
            model_dump = file.read()
            self.loss.load(model_dump)

    def save_to_file(self, file_name):
        with open(file_name, "w") as file:
            model_dump = self.loss.save()
            file.write(model_dump)

    # For a set of next step boards and coords of next step
    # calculates value and stores it in the coords of next step.
    def get_next_step_values(self, boards):
        values = copy.deepcopy(START_VALUES)
        for bxy in boards:
            b, x, y = bxy
            self.x.set(b)
            value = self.prediction.val()
            values[x][y] = value[0][0]
        return values

    def apply_gradient(self):

        norm = lambda matrix: math.sqrt(sum(sum(x**2 for x in row) for row in matrix))

        alpha = 0.01
        # TODO: interesting! explore more
        # w1norm = norm(self.w1.val())
        # w1dnorm = norm(self.w1.dval())
        # alpha = 0.01
        # if w1dnorm / w1norm < 0.01:
        #    alpha = 0.1
        # if w1dnorm / w1norm > 1000:
        #    alpha = 0.001
        # alpha = 0.0001

        self.w1.appl(alpha)
        self.b1.appl(alpha)
        self.w2.appl(alpha)
        self.b2.appl(alpha)
        # self.w3.appl(alpha)
        # self.b3.appl(alpha)
