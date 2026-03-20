from __future__ import annotations
import random
import copy
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import math
from listinvert import Matrix, value as mx_value

START_BOARD = [[0 for _ in range(6)] for _ in range(6)]

MCTS_NUM_SIMULATIONS = 1000


class GameType(Enum):
    TICTACTOE_6_6_4 = 1
    TICTACTOE_6_6_5_TOR = 2


def make_matrix_board(cells: list[list[int]], game_type: GameType = GameType.TICTACTOE_6_6_4) -> 'Board':
    m = Matrix(6, 6)
    m.set_data(cells)
    return Board(m, game_type=game_type)


class Board:
    cells: Matrix
    game_type: GameType

    def __init__(self, board: Optional[Matrix] = None, game_type: GameType = GameType.TICTACTOE_6_6_4) -> None:
        if board is None:
            self.cells = Matrix(6, 6)
            self.reset()
        else:
            self.cells = Matrix(board)
        self.game_type = game_type

    def reset(self) -> None:
        self.cells.set_data(START_BOARD)

    def set(self, board: Matrix) -> None:
        if board.rows != 6 or board.cols != 6:
            raise ValueError("board must be 6x6")
        self.cells = Matrix(board)

    def copy(self) -> Board:
        return Board(Matrix(self.cells), self.game_type)

    def asstr(self) -> str:
        s = ""
        for row in range(6):
            for col in range(6):
                s += str(int(self.cells.get(row, col)))
        return s

    # Generates all boards for next single step
    # Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
    def all_next_steps(self, next_player: int) -> list[tuple[Board, int, int]]:
        boards = []
        for row in range(6):
            for col in range(6):
                if self.cells.get(row, col) == 0:
                    next_board = self.copy()  # copy.deepcopy(board)
                    next_board.cells.set(row, col, next_player)
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
        b = self.cells

        lll = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 2), (3, 3)],
            [(-1, 1), (-2, 2), (-3, 3)],
        ]

        g = lambda x, y: b.get(x, y) if -1 < x < 6 and -1 < y < 6 else None

        xyo: list[tuple[int, int]] = []
        winner: Optional[int] = None
        there_are_empty_cells = False
        for i in range(6):
            for j in range(6):
                cur = b.get(i, j)
                if cur == 0:
                    there_are_empty_cells = True
                    continue

                for ll in lll:
                    xy = [(i + lx, j + ly) for lx, ly in ll]
                    if all([g(x, y) == cur for x, y in xy]):
                        if winner is not None and winner != cur:
                            return None, []
                        winner = int(cur)
                        xyo = xyo + [(i, j)] + xy
        if winner is None and not there_are_empty_cells:
            winner = 0

        return winner, sorted(set(xyo))

    def check_winner_tictactoe_6_6_5_tor(self) -> tuple[Optional[int], list[tuple[int, int]]]:
        b = self.cells

        lll = [
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        ]

        g = lambda x, y: b.get(x, y) if -1 < x < 6 and -1 < y < 6 else None

        xyo: list[tuple[int, int]] = []
        winner: Optional[int] = None
        there_are_empty_cells = False
        for i in range(6):
            for j in range(6):
                cur = b.get(i, j)
                if cur == 0:
                    there_are_empty_cells = True
                    continue

                for ll in lll:
                    xy = [((i + lx) % 6, (j + ly) % 6) for lx, ly in ll]
                    if all([g(x, y) == cur for x, y in xy]):
                        if winner is not None and winner != cur:
                            return None, []  # double winners, wrong
                        winner = int(cur)
                        xyo = xyo + [(i, j)] + xy
        if winner is None and not there_are_empty_cells:
            winner = 0
        return winner, sorted(set(xyo))



#@dataclass
class GameState:
    board: Board
    next_player: int  # 1 for crosses, -1 for zeroes
    prev_move: Optional[tuple[int, int]]  # coordinates of last move
    turn_number: int
    winner: Optional[int] = None
    winning_row: Optional[list[tuple[int, int]]] = None  # if the state is terminal, contains list of winning cells
    reward: Optional[list[list[float]]] = None
    policy: Optional[list[list[float]]] = None

    def almost_equal(self, other: GameState, delta: float = 0.001) -> bool:
        if self.board.cells != other.board.cells:
            return False
        if self.reward is None and other.reward is None:
            return True
        if self.reward is None or other.reward is None:
            return False
        # compare reward elements
        for r1, r2 in zip(self.reward, other.reward):
            for v1, v2 in zip(r1, r2):
                if abs(v1 - v2) > delta:
                    return False
        return True

    def __init__(self, board: Board, next_player: int, turn_number: int = 0, prev_move: Optional[tuple[int, int]] = None, reward: Optional[list[list[float]]] = None, policy: Optional[list[list[float]]] = None, winner: Optional[int] = None, winning_row: Optional[list[tuple[int, int]]] = None) -> None:
        self.board = board
        self.next_player = next_player
        self.turn_number = turn_number
        self.prev_move = prev_move
        self.reward = reward
        self.policy = policy
        self.winner = winner
        self.winning_row = winning_row

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

        winner, winning_row = self.winner, self.winning_row or []

        print("Step", self.turn_number, ":", "crosses" if self.next_player == -1 else "zeroes")
        x, y = self.prev_move if self.prev_move else (-1, -1)
        print("  Move:", x, y, " Reward: ", self.reward)

        for i in range(6):
            for j in range(6):
                cell = self.board.cells.get(i, j)

                bg = "grey" if (i + j) % 2 == 0 else "black"
                if cell == -1:
                    what, fg = " O ", "green"
                elif cell == 1:
                    what, fg = " X ", "blue"
                else:
                    what, fg = "   ", "std"
                fg = "red" if (i, j) in winning_row else fg

                if i == x and j == y:
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

    def best_greedy_step(self, prev_state: GameState) -> GameState:
        board = prev_state.board.copy()
        next_player = prev_state.next_player

        boards = board.all_next_steps(next_player)
        if len(boards) == 0:
            return prev_state

        best = -100.0
        best_xy = (-1, -1)
        m = self.model_x if next_player == 1 else self.model_o
        
        greedy_policy: list[list[float]] = [[0.0 for _ in range(6)] for _ in range(6)]

        for next_board, row, col in boards:
            value = m.get_next_step_value(next_player, next_board.cells)
            if value is None:
                continue

            greedy_policy[row][col] = value

            if value > best:
                best = value
                best_xy = (row, col)

        row, col = best_xy
        board.cells.set(row, col, next_player)
        winner, winning_row = board.check_winner()

        return GameState(
                board=board, 
                next_player=-next_player, 
                turn_number=prev_state.turn_number + 1,
                prev_move=(row, col),
                policy=greedy_policy,
                winner=winner,
                winning_row=winning_row)



    def random_step(self, prev_state: GameState) -> GameState:
        board = prev_state.board.copy()
        next_player = prev_state.next_player
        empty_cells = [(r, c) for r in range(6) for c in range(6) if board.cells.get(r, c) == 0]
        row, col = random.choice(empty_cells)

        board.cells.set(row, col, next_player)
        winner, winning_row = board.check_winner()

        return GameState(
                board=board, 
                next_player=-next_player, 
                turn_number=prev_state.turn_number + 1,
                prev_move=(row, col), 
                winner=winner,
                winning_row=winning_row)

    def turn_number(self, board: Board) -> int:
        # GameState number is count of O's on the board.
        return sum([1 for r in range(6) for c in range(6) if board.cells.get(r, c) == -1])

    def choose_next_step(self, prev_state: GameState) -> GameState:
        # First step is always random to increase diversity
        if self.turn_number(prev_state.board) == 0:
            return self.random_step(prev_state)
        elif self.game_mode == "mcts":
            from lib.mcts import best_mcts_step
            return best_mcts_step(self, prev_state, MCTS_NUM_SIMULATIONS)
        else:
            return self.best_greedy_step(prev_state)

    # Returns list of consequtive game states
    # The reward of last state shows the game winner
    def play_game(self, start_board: Optional[Board] = None) -> list[GameState]:
        if start_board is None:
            start_board = Board(game_type=self.game_type)
        steps = []

        init_state = GameState(board=start_board.copy(), next_player=1, prev_move=None, turn_number=0)
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

        return steps

    def competition(self, num_games: int) -> dict[int, int]:
      winners = {-1: 0, 0: 0, 1: 0}
      for f in range(num_games):
        steps = self.play_game()
        assert steps[-1].reward is not None
        winner = steps[-1].reward[0][0]
        winners[int(winner)] = winners[int(winner)] + 1

      return winners

    def generate_batch_from_games(self, num_boards: Optional[int] = None, shuffle: bool = True, num_games: Optional[int] = None) -> list[GameState]:
        all_steps: list[GameState] = []

        if num_boards is not None:
            if num_games is not None:
                raise ValueError("Exactly one of num_boards or num_games must be provided")
            while len(all_steps) < num_boards:
                all_steps.extend(self.play_game())
        else:
            if num_games is None:
                raise ValueError("Exactly one of num_boards or num_games must be provided")
            for _ in range(num_games):
                all_steps.extend(self.play_game())

        if shuffle:
            random.shuffle(all_steps)
        return [step for step in all_steps if step.reward is not None]
    
