import random
import copy
from dataclasses import dataclass
from typing import Optional
from enum import Enum

START_BOARD = [[0 for _ in range(6)] for _ in range(6)]

DEFAULT_VALUES = [[0 for _ in range(6)] for _ in range(6)]


class GameType(Enum):
    TICTACTOE_6_6_4 = 1
    TICTACTOE_6_6_5_TOR = 2

class Board:
    # values: list[list[float]]

    def __init__(self, board=None, game_type=GameType.TICTACTOE_6_6_4):
        if not board:
            self.reset()
        else:
            self.set(board)
        self.game_type = game_type

    def reset(self):
        self.board = copy.deepcopy(START_BOARD)

    def set(self, board):
        self.board = board

    def copy(self):
        return Board(copy.deepcopy(self.board), self.game_type)

    # Generates all boards for next single step (ply=1 crosses, ply=-1 zeroes)
    # Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
    def all_next_steps(self, ply):
        boards = []
        for row in range(6):
            for col in range(6):
                if self.board[row][col] == 0:
                    next_board = self.copy()  # copy.deepcopy(board)
                    next_board.board[row][col] = ply
                    boards.append((next_board, row, col))
        return boards

    # Returns 1 if crosses win, -1 if zeroes win, 0 if tie,
    # and None if board is invalid
    def check_winner(self):
        if self.game_type==GameType.TICTACTOE_6_6_4:
            return self.check_winner_tictactoe_6_6_4()
        else:
            return self.check_winner_tictactoe_6_6_5_tor()

    def check_winner_tictactoe_6_6_4(self):
        b = self.board

        lll = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 2), (3, 3)],
            [(-1, 1), (-2, 2), (-3, 3)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo = []
        winner = 0
        for i in range(6):
            for j in range(6):
                if b[i][j] == 0:
                    continue

                for ll in lll:
                    xy = [(i + lx, j + ly) for lx, ly in ll]
                    if all([g(x, y) == b[i][j] for x, y in xy]):
                        if winner != 0 and winner != b[i][j]:
                            return None, []
                        winner = b[i][j]
                        xyo = xyo + [(i, j)] + xy

        return winner, sorted(set(xyo))


    def check_winner_tictactoe_6_6_5_tor(self):
        b = self.board

        lll = [
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo = []
        winner = 0
        for i in range(6):
            for j in range(6):
                cur = b[i][j]
                if cur == 0:
                    continue

                for ll in lll:
                    xy = [( (i + lx)%6, (j + ly)%6 ) for lx, ly in ll]
                    if all([g(x, y) == cur for x, y in xy]):
                        if winner != 0 and winner != cur:
                            return None, [] # double winners, wrong
                        winner = cur
                        xyo = xyo + [(i, j)] + xy

        return winner, sorted(set(xyo))



    def print_board(self):

        bgs = {"grey": "\033[100m", "black": "\033[40m"}

        fgs = {
            "green": "\033[32m",
            "blue": "\033[94m",
            "red": "\033[31m",
        }

        cancel_color = "\033[0m"

        def cprint(fg, bg, what):
            if bg in bgs:
                what = bgs[bg] + what + cancel_color
            if fg in fgs:
                what = fgs[fg] + what + cancel_color
            print(what, end="")

        winner, xyo = self.check_winner()

        for i, row in enumerate(self.board):
            for j, cell in enumerate(row):

                bg = "grey" if (i + j) % 2 == 0 else "black"
                if cell == -1:
                    what, fg = " O ", "green"
                elif cell == 1:
                    what, fg = " X ", "blue"
                else:
                    what, fg = "   ", "std"
                fg = "red" if (i, j) in xyo else fg
                cprint(fg, bg, what)

            print()


@dataclass
class Step:
    step_no: int
    ply: int  # 1 for crosses, -1 for zeroes
    x: int
    y: int
    board: list[list[int]]
    values: list[list[float]]
    reward: Optional[float] = None


class Game:
    def __init__(self, model_x, model_o, game_type=GameType.TICTACTOE_6_6_4, game_mode="greedy"):
        self.game_type = game_type
        self.game_mode = game_mode
        self.model_x = model_x
        self.model_o = model_o
        self.board = Board(game_type=self.game_type)

    def best_greedy_step(self, board, ply):

        boards = board.all_next_steps(ply)
        if len(boards) == 0:
            return None, None, None

        best = -100 if ply == 1 else 100
        best_xy = (-1, -1)
        m = self.model_x if ply == 1 else self.model_o
        
        values = copy.deepcopy(DEFAULT_VALUES) 
        for board, row, col in boards:
            value = m.get_next_step_value(board.board)
            values[row][col] = value
            if value is None:
                continue

            if ply == 1 and value > best:
                best = value
                best_xy = (row, col)
            if ply == -1 and value < best:
                best = value
                best_xy = (row, col)

        return best_xy[0], best_xy[1], values


    def best_minimax_step(self, board, ply):
        print("NOT IMPLEMENTED")
        return self.best_greedy_step(board, ply)


    def random_step(self):
        cell = random.randint(0, 36)
        return int(cell / 6), int(cell % 6), copy.deepcopy(DEFAULT_VALUES) 


    def make_next_step(self, ply, step_no):
        if self.game_mode == "minimax":
           x, y, values = self.best_minimax_step(self.board, ply, step_no)
        else:
           # First step is always random to increase diversity
           if step_no == 0: 
              return self.random_step()

           x, y, values = self.best_greedy_step(self.board, ply)

        return x, y, values


    def play_game(self):
        self.board.reset()
        steps, step_no, ply, winner = [], 0, 1, 0
        while True:
            x, y, values = self.make_next_step(ply, step_no)
            if x is None and y is None:
                break  ## the board is full, no more steps
            self.board.board[x][y] = ply

            ss = Step(
                step_no=step_no,
                ply=ply,
                x=x,
                y=y,
                board=self.board.copy(),
                values=copy.deepcopy(values),
            )
            steps.append(ss)

            winner, _ = self.board.check_winner()
            if winner != 0:
                break

            ply = -ply
            if ply == 1:
                step_no = step_no + 1

        # Set desired rewards to the boards
        reward = winner
        for step in reversed(steps):
            step.reward = reward
            reward = reward * 0.9

        return steps, winner


# ----------------------------------
def generate_random_board():
    size = 6 * 6
    num_zeroes = random.randint(
        0, size // 2
    )  # Random number of zeroes (up to half the board)
    num_crosses = num_zeroes + random.choice([0, 1])  # Either equal or one more cross
    num_empty = size - num_zeroes - num_crosses

    values = [1] * num_crosses + [-1] * num_zeroes + [0] * num_empty
    random.shuffle(values)

    return [values[i * 6 : (i + 1) * 6] for i in range(6)]


# Generates a random batch of size N, where each class is presented with n // 3 samples
def generate_batch(n):
    boards, winners = [], []
    for board_class in range(-1, 2):
        for i in range(n // 3):
            while True:
                board = generate_random_board()
                winner, _ = Board(board).check_winner()
                if winner == board_class:
                    break
            boards.append(board)
            winners.append([(winner + 1.0) / 2.0])
    return boards, winners


# Generate sequence of boards for a single random game
# TODO: delete, use play_game with TTTRandom models
def generate_random_game():
    boards = [START_BOARD]

    ply = 1  # crosses
    num = 0  # number of filled cells
    while num < 36:
        row = random.randint(0, 5)
        col = random.randint(0, 5)

        if board[row][col] == 0:
            board[row][col] = ply
            num = num + 1
            ply = -ply

            boards.append(copy.deepcopy(board))

            winner, _ = check_winner(board)
            if winner != 0:
                return boards, winner

    return boards, 0


def print_scores(values):
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            bg_color = "\033[100m" if (i + j) % 2 == 0 else "\033[40m"
            score = round(value * 100) if value is not None else "  "
            if value is not None:
                score = round(value * 100)
                print(bg_color + f" {score:02}" + "\033[0m", end="")
            else:
                print(bg_color + f"   " + "\033[0m", end="")

        print()







def competition(model_x, model_o, num_games, game_type=GameType.TICTACTOE_6_6_4):
    winners = {-1: 0, 0: 0, 1: 0}
    g = Game(model_x, model_o, game_type)
    for f in range(num_games):
        _, winner = g.play_game()
        winners[winner] = winners[winner] + 1

    return winners
