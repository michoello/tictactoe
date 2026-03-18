from __future__ import annotations
import random
import copy
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import math

START_BOARD = [[0 for _ in range(6)] for _ in range(6)]

MCTS_NUM_SIMULATIONS = 1000


class GameType(Enum):
    TICTACTOE_6_6_4 = 1
    TICTACTOE_6_6_5_TOR = 2


class Board:
    state: list[list[int]]
    game_type: GameType

    def __init__(self, board: Optional[list[list[int]]] = None, game_type: GameType = GameType.TICTACTOE_6_6_4) -> None:
        if not board:
            self.reset()
        else:
            self.set(board)
        self.game_type = game_type

    def reset(self) -> None:
        self.state = copy.deepcopy(START_BOARD)

    def set(self, board: list[list[int]]) -> None:
        if len(board) != 6:
            raise ValueError("board must have 6 rows sharp")
        for row in board:
            if len(row) != 6:
                raise ValueError("each row must have 6 cols sharp")
        self.state = board

    def copy(self) -> Board:
        return Board(copy.deepcopy(self.state), self.game_type)

    def asstr(self) -> str:
        s = ""
        for row in range(6):
            for col in range(6):
                s += str(self.state[row][col])
        return s

    # Generates all boards for next single step (last_move=1 crosses, last_move=-1 zeroes)
    # Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
    def all_next_steps(self, last_move: int) -> list[tuple[Board, int, int]]:
        boards = []
        for row in range(6):
            for col in range(6):
                if self.state[row][col] == 0:
                    next_board = self.copy()  # copy.deepcopy(board)
                    next_board.state[row][col] = last_move
                    boards.append((next_board, row, col))
        return boards

    # Returns 1 if crosses win, -1 if zeroes win, 0 if tie,
    # and None if board is invalid
    def check_winner(self) -> tuple[Optional[int], list[tuple[int, int]]]:
        if self.game_type == GameType.TICTACTOE_6_6_4:
            return self.check_winner_tictactoe_6_6_4()
        else:
            return self.check_winner_tictactoe_6_6_5_tor()

    def check_winner_tictactoe_6_6_4(self) -> tuple[Optional[int], list[tuple[int, int]]]:
        b = self.state

        lll = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 2), (3, 3)],
            [(-1, 1), (-2, 2), (-3, 3)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo: list[tuple[int, int]] = []
        winner: Optional[int] = None
        there_are_empty_cells = False
        for i in range(6):
            for j in range(6):
                if b[i][j] == 0:
                    there_are_empty_cells = True
                    continue

                for ll in lll:
                    xy = [(i + lx, j + ly) for lx, ly in ll]
                    if all([g(x, y) == b[i][j] for x, y in xy]):
                        if winner is not None and winner != b[i][j]:
                            return None, []
                        winner = b[i][j]
                        xyo = xyo + [(i, j)] + xy
        if winner is None and not there_are_empty_cells:
            winner = 0

        return winner, sorted(set(xyo))

    def check_winner_tictactoe_6_6_5_tor(self) -> tuple[Optional[int], list[tuple[int, int]]]:
        b = self.state

        lll = [
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo: list[tuple[int, int]] = []
        winner: Optional[int] = None
        there_are_empty_cells = False
        for i in range(6):
            for j in range(6):
                cur = b[i][j]
                if cur == 0:
                    there_are_empty_cells = True
                    continue

                for ll in lll:
                    xy = [((i + lx) % 6, (j + ly) % 6) for lx, ly in ll]
                    if all([g(x, y) == cur for x, y in xy]):
                        if winner is not None and winner != cur:
                            return None, []  # double winners, wrong
                        winner = cur
                        xyo = xyo + [(i, j)] + xy
        if winner is None and not there_are_empty_cells:
            winner = 0
        return winner, sorted(set(xyo))



#@dataclass
class GameState:
    board: Board
    last_move: int  # 1 for crosses, -1 for zeroes
    x: int  # coordinates of last move
    y: int
    step_no: int
    winner: Optional[int] = None
    xyo: Optional[list[tuple[int, int]]] = None  # if the state is terminal, contains list of winning cells
    reward: Optional[list[list[float]]] = None

    def __init__(self, board: Board, last_move: int, step_no: int = 0, x: int = -1, y: int = -1) -> None:
        self.board = board
        self.last_move = last_move
        self.step_no = step_no
        self.x = x
        self.y = y

    def print_state(self) -> None:
        bgs = {
            # 256colors
            # bash for i in {0..255}; do printf "\033[48;5;%sm %3d \033[0m" "$i" "$i"; done; echo
            "grey": "\033[48;5;236m",
            "black": "\033[48;5;234m",
            "yellow": "\033[38;5;94m",
        }

        fgs = {
            "green": "\033[32m",
            "blue": "\033[36m",
            "red": "\033[31m",
            "yellow": "\033[33m",
        }

        cancel_color = "\033[0m"

        def cprint(fg: str, bg: str, what: str) -> None:
            if bg in bgs:
                what = bgs[bg] + what + cancel_color
            if fg in fgs:
                what = fgs[fg] + what + cancel_color
            print(what, end="")

        winner, xyo = self.winner, self.xyo or []

        print("Step", self.step_no, ":", "crosses" if self.last_move == 1 else "zeroes")
        print("  Move:", self.x, self.y, " Reward: ", self.reward)

        for i in range(6):
            for j in range(6):
                cell = self.board.state[i][j]

                bg = "grey" if (i + j) % 2 == 0 else "black"
                if cell == -1:
                    what, fg = " O ", "green"
                elif cell == 1:
                    what, fg = " X ", "blue"
                else:
                    what, fg = "   ", "std"
                fg = "red" if (i, j) in xyo else fg

                if i == self.x and j == self.y:
                    bg = "yellow"

                cprint(fg, bg, what)

            print()

# Entire MCTS algorithm is in this class
from lib.mcts import MctsNode

from typing import Any
class Game:
    game_type: GameType
    game_mode: str
    model_x: Any
    model_o: Any
    
    def __init__(
        self, model_x: Any, model_o: Any, game_type: GameType = GameType.TICTACTOE_6_6_4, game_mode: str = "greedy"
    ) -> None:
        self.game_type = game_type
        self.game_mode = game_mode
        self.model_x = model_x
        self.model_o = model_o

    # Returns coordinates of next step and the policy 6*6 matrix
    # Args: board, next_move is 1 for Xs, -1 for Os
    def best_greedy_step(self, board: Board, next_move: int) -> tuple[Optional[int], Optional[int], list[list[float]]]:

        boards = board.all_next_steps(next_move)
        if len(boards) == 0:
            return None, None, []

        #best = -100 if next_move == 1 else 100
        best = -100.0
        best_xy = (-1, -1)
        m = self.model_x if next_move == 1 else self.model_o
        
        greedy_policy: list[list[float]] = [[0.0 for _ in range(6)] for _ in range(6)]

        for board, row, col in boards:
            value = m.get_next_step_value(next_move, board.state)
            if value is None:
                continue

            greedy_policy[row][col] = value

            if value > best:
                best = value
                best_xy = (row, col)

        return best_xy[0], best_xy[1], greedy_policy



    def random_step(self) -> tuple[int, int]:
        cell = random.randint(0, 35)
        return int(cell / 6), int(cell % 6)

    def step_no(self, board: Board) -> int:
        # GameState number is count of O's on the board.
        return sum([1 for row in board.state for x in row if x == -1])

    def choose_next_step(self, prev_state: GameState) -> GameState:
        board = prev_state.board.copy()
        next_move = -prev_state.last_move if prev_state.last_move is not None else 1

        # First step is always random to increase diversity
        row: Optional[int] = None
        col: Optional[int] = None
        if self.step_no(board) == 0:
            row, col = self.random_step()
        elif self.game_mode == "mcts":
            game_state = GameState(
                board=board, 
                last_move=next_move, 
                step_no=prev_state.step_no + 1)

            from lib.mcts import best_mcts_step
            row, col = best_mcts_step(self, game_state, MCTS_NUM_SIMULATIONS)
            return game_state

        else:
            row, col, policy = self.best_greedy_step(board, next_move)


        game_state = GameState(
                board=board, 
                last_move=next_move, 
                step_no=prev_state.step_no + 1)

        assert row is not None and col is not None
        game_state.x=row 
        game_state.y=col

        board.state[row][col] = next_move
        winner, xyo = board.check_winner()

        game_state.winner=winner
        game_state.xyo=xyo
        return game_state

    # Returns list of consequtive game states
    # The reward of last state shows the game winner
    def play_game(self, start_board: Optional[Board] = None) -> list[GameState]:
        if start_board is None:
            start_board = Board(game_type=self.game_type)
        steps = []

        init_state = GameState(board=start_board.copy(), last_move=-1, x=-1, y=-1, step_no=0)
        steps.append(init_state)
        while steps[-1].winner is None:
            prev_state = steps[-1]
            next_state = self.choose_next_step(prev_state)
            steps.append(next_state)

        # Set desired rewards to the boards
        assert steps[-1].winner is not None
        reward: float = float(steps[-1].winner)
        for step in reversed(steps):
            step.reward = [[reward]]
            reward = reward * 0.9

        # lil trick to remove initial empty state. Is it better? Or should it stay?
        steps.pop(0)

        return steps

    def competition(self, num_games: int) -> dict[int, int]:
      winners = {-1: 0, 0: 0, 1: 0}
      for f in range(num_games):
        steps = self.play_game()
        assert steps[-1].reward is not None
        winner = steps[-1].reward[0][0]
        winners[int(winner)] = winners[int(winner)] + 1

      return winners

    def generate_batch_from_games(self, num_boards: int, shuffle: bool = True) -> list['GameState']:
        all_steps: list[GameState] = []
        while len(all_steps) < num_boards:
            all_steps.extend(self.play_game())

        if shuffle:
            random.shuffle(all_steps)
        return [step for step in all_steps if step.reward is not None]
    
