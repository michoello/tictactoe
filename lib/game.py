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
    # values: list[list[float]]

    def __init__(self, board=None, game_type=GameType.TICTACTOE_6_6_4):
        if not board:
            self.reset()
        else:
            self.set(board)
        self.game_type = game_type

    def reset(self):
        self.state = copy.deepcopy(START_BOARD)

    def set(self, board):
        if len(board) != 6:
            raise ValueError("board must have 6 rows sharp")
        for row in board:
            if len(row) != 6:
                raise ValueError("each row must have 6 cols sharp")
        self.state = board

    def copy(self):
        return Board(copy.deepcopy(self.state), self.game_type)

    def asstr(self):
        s = ""
        for row in range(6):
            for col in range(6):
                s += str(self.state[row][col])
        return s

    # Generates all boards for next single step (last_move=1 crosses, last_move=-1 zeroes)
    # Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
    def all_next_steps(self, last_move):
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
    def check_winner(self):
        if self.game_type == GameType.TICTACTOE_6_6_4:
            return self.check_winner_tictactoe_6_6_4()
        else:
            return self.check_winner_tictactoe_6_6_5_tor()

    def check_winner_tictactoe_6_6_4(self):
        b = self.state

        lll = [
            [(0, 1), (0, 2), (0, 3)],
            [(1, 0), (2, 0), (3, 0)],
            [(1, 1), (2, 2), (3, 3)],
            [(-1, 1), (-2, 2), (-3, 3)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo = []
        winner = None
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

    def check_winner_tictactoe_6_6_5_tor(self):
        b = self.state

        lll = [
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 0), (2, 0), (3, 0), (4, 0)],
            [(1, 1), (2, 2), (3, 3), (4, 4)],
            [(-1, 1), (-2, 2), (-3, 3), (-4, 4)],
        ]

        g = lambda x, y: b[x][y] if -1 < x < 6 and -1 < y < 6 else None

        xyo = []
        winner = None
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



@dataclass
class GameState:
    board: Board
    last_move: int  # 1 for crosses, -1 for zeroes
    x: int  # coordinates of last move
    y: int
    step_no: int
    winner: Optional[int] = None
    xyo: Optional[list[int]] = None  # if the state is terminal, contains list of winning cells
    reward: Optional[float] = None

    def print_state(self):
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

        def cprint(fg, bg, what):
            if bg in bgs:
                what = bgs[bg] + what + cancel_color
            if fg in fgs:
                what = fgs[fg] + what + cancel_color
            print(what, end="")

        #winner, xyo = self.board.check_winner()
        winner, xyo = self.winner, self.xyo

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
class MctsNode:
    parent = None  # previous step node, None for root
    board: Board
    row: int
    col: int
    next_move: int  ## 1 or -1
    all_moves: list[Board] = []
    tried_nodes: list[Board] = []
    state_value: float = 0  # state value taken from model
    value: float = 0.0  # accumulated value collected from child nodes
    num_visits: int = 0  # number of times simulation passed through the node
    prior: float = 0.0
    is_terminal = False

    def __init__(self, board, row, col, next_move):
        self.board = board
        self.row = row
        self.col = col
        self.next_move = next_move
        self.num_visits = 0
        self.all_moves = self.board.all_next_steps(self.next_move)
        self.tried_nodes = []

    def mcts_get_best_node_to_continue(self):
        n_total = self.num_visits
        n_sqrt_total = math.sqrt(n_total)
        c_puct = 1.0  # TODO: experiment with that
        best_puct = -1000
        best_node = None
        for node in self.tried_nodes:
            value = node.value
            if self.next_move == -1:
                value = -value
            node_puct = (
                value / node.num_visits + node.prior * n_sqrt_total / node.num_visits
            )
            if node_puct > best_puct:
                best_puct = node_puct
                best_node = node

        return best_node

    # Currently the values are taken from the model value outputs, so they are not normalized
    # Therefore have to softmax them for puct to work
    # TODO: make a policy network and take those priors from there
    def mcts_softmax_priors(self):
        next_move = self.next_move
        tried_nodes = self.tried_nodes
        # converting state values back to [0;1] range, and inverting them to 1-v for Os
        # TODO: all these ugly tricks should go away
        norm_values = [node.state_value for node in tried_nodes]
        norm_values = [(v + 1) / 2.0 for v in norm_values]
        if next_move == -1:
            norm_values = [1 - v for v in norm_values]
        priors = norm_values

        # softmax
        m = max(priors)
        exps = [math.exp(v - m) for v in priors]
        total = sum(exps)
        priors = [e / total for e in exps]

        for i, node in enumerate(tried_nodes):
            node.prior = priors[i]

    # Returns the last visited node
    def mcts_run_simulation(self, gm):  ## `gm` is game object
        if self.is_terminal:
            # no more walking after terminal
            return self

        tried, all = len(self.tried_nodes), len(self.all_moves)
        if tried < all:
            board, row, col = self.all_moves[tried]

            next_node = MctsNode(board, row, col, -self.next_move)

            winner, _ = board.check_winner()
            if winner is not None:
                next_node.state_value = winner
                next_node.is_terminal = True
            else:
                m = gm.model_x if next_node.next_move == 1 else gm.model_o
                next_node.state_value = m.get_next_step_value(board.state)
                # Applying this ugly patch to make it range [-1;1]
                # as currently model returns [0;1]
                next_node.state_value = (next_node.state_value * 2) - 1
                # Collecting all values is too slow. TODO: policy output
                # brds = [(board.state, r, c) for board, r, c in self.all_moves]
                # m.get_next_step_values(brds)
            next_node.parent = self

            self.tried_nodes.append(next_node)

            if tried == all - 1:  # we just added last move, time to calc priors
                self.mcts_softmax_priors()

            return next_node

        next_node = self.mcts_get_best_node_to_continue()
        return next_node.mcts_run_simulation(gm)

    def mcts_back_propagate(self):
        cur_node = self
        while cur_node is not None:
            cur_node.num_visits += 1
            cur_node.value += self.state_value
            cur_node = cur_node.parent

    def mcts_node_count(self):
        count = 0
        for node in self.tried_nodes:
            count += node.mcts_node_count()
        return count + 1

    def mcts_depth(self):
        depth = 0
        if len(self.tried_nodes) == 0:
            return 1

        for node in self.tried_nodes:
            subdepth = node.mcts_depth()
            if subdepth > depth:
                depth = subdepth
        return depth + 1

    def mcts_unique_node_count(self, accum=None):
        if accum is None:
            accum = {}
        hsh = self.board.asstr()
        accum[hsh] = 1

        for node in self.tried_nodes:
            node.mcts_unique_node_count(accum)
        return len(accum.keys())

    def mcts_terminal_node_count(self):
        if self.is_terminal:
            return 1
        cnt = 0
        for node in self.tried_nodes:
            cnt += node.mcts_terminal_node_count()
        return cnt

    def mcts_analyze_tree(self):
        print("MCTS available moves: ", len(self.all_moves))
        print("MCTS node count: ", self.mcts_node_count())
        print("MCTS unique count: ", self.mcts_unique_node_count())
        print("MCTS terminal count: ", self.mcts_terminal_node_count())
        print("MCTS tree depth: ", self.mcts_depth())
        print()


class Game:
    def __init__(
        self, model_x, model_o, game_type=GameType.TICTACTOE_6_6_4, game_mode="greedy"
    ):
        self.game_type = game_type
        self.game_mode = game_mode
        self.model_x = model_x
        self.model_o = model_o

    def best_greedy_step(self, board, next_move):

        boards = board.all_next_steps(next_move)
        if len(boards) == 0:
            return None, None, None

        best = -100 if next_move == 1 else 100
        best_xy = (-1, -1)
        m = self.model_x if next_move == 1 else self.model_o

        for board, row, col in boards:
            value = m.get_next_step_value(board.state)
            if value is None:
                continue

            if next_move == 1 and value > best:
                best = value
                best_xy = (row, col)
            if next_move == -1 and value < best:
                best = value
                best_xy = (row, col)

        return best_xy[0], best_xy[1]

    def best_minimax_step(self, board, next_move):
        player = "X" if next_move == 1 else "O"
        depth = 2
        alpha, beta = -1000, 1000  # infinity!
        val, row, col = self.minimax(board, depth, alpha, beta, player)
        return row, col

    def best_mcts_step(self, board, next_move, num_simulations):
        root = MctsNode(board, None, None, next_move)

        for sim_num in range(num_simulations):
            new_leaf_node = root.mcts_run_simulation(self)
            new_leaf_node.mcts_back_propagate()

        root.mcts_analyze_tree()

        # Choose the node that got the most visits
        best_node = None
        best_count = -1
        for node in root.tried_nodes:
            if node.num_visits > best_count:
                best_count = node.num_visits
                best_node = node

        return best_node.row, best_node.col

    def minimax(self, board, depth, alpha, beta, player):
        winner, _ = board.check_winner()
        if winner is not None:
            return (winner, None, None)

        m = self.model_x if player == "X" else self.model_o
        if depth == 0:
            return m.get_next_step_value(board.state), None, None

        boards = board.all_next_steps(1 if player == "X" else -1)  # next_move)
        random.shuffle(boards)

        best_val = -float("inf") if player == "X" else float("inf")
        next_player = "O" if player == "X" else "X"
        best_row, best_col = None, None

        for board, row, col in boards:
            val, _, _ = self.minimax(board, depth - 1, alpha, beta, next_player)
            if player == "X":
                if val > best_val:
                    best_val, best_row, best_col = val, row, col
                alpha = max(alpha, best_val)
            else:
                if val < best_val:
                    best_val, best_row, best_col = val, row, col
                beta = min(beta, best_val)

            if beta <= alpha:
                break

        return best_val, best_row, best_col

    def random_step(self):
        cell = random.randint(0, 35)
        return int(cell / 6), int(cell % 6)

    def step_no(self, board):
        # GameState number is count of O's on the board.
        return sum([1 for row in board.state for x in row if x == -1])

    def choose_next_step(self, prev_state):
        board = prev_state.board.copy()
        next_move = -prev_state.last_move
        # First step is always random to increase diversity
        row, col = None, None
        if self.step_no(board) == 0:
            row, col = self.random_step()
        elif self.game_mode == "minimax":
            row, col = self.best_minimax_step(board, next_move)
        elif self.game_mode == "mcts":
            row, col = self.best_mcts_step(board, next_move, MCTS_NUM_SIMULATIONS)
        else:
            row, col = self.best_greedy_step(board, next_move)

        board.state[row][col] = next_move
        winner, xyo = board.check_winner()

        return GameState(
                board=board, 
                last_move=next_move, 
                x=row, 
                y=col, 
                step_no=prev_state.step_no + 1, 
                winner=winner,
                xyo=xyo)

    # Returns list of consequtive game states
    # The reward of last state shows the game winner
    def play_game(self):
        board = Board(game_type=self.game_type)
        steps = []

        init_state = GameState(board=board.copy(), last_move=-1, x=None, y=None, step_no=0)
        steps.append(init_state)
        while steps[-1].winner is None:
            prev_state = steps[-1]
            next_state = self.choose_next_step(prev_state)
            steps.append(next_state)

        # Set desired rewards to the boards
        reward = steps[-1].winner
        for step in reversed(steps):
            step.reward = reward
            reward = reward * 0.9

        # lil trick to remove initial empty state. Is it better? Or should it stay?
        steps.pop(0)

        return steps

    def competition(self, num_games):
      winners = {-1: 0, 0: 0, 1: 0}
      for f in range(num_games):
        steps = self.play_game()
        winner = steps[-1].reward
        winners[winner] = winners[winner] + 1

      return winners

    def generate_batch_from_games(self, num_boards):
        all_steps = []
        while len(all_steps) < num_boards:
            all_steps.extend(self.play_game())

        random.shuffle(all_steps)
        return [step.board.state for step in all_steps], [ [(step.reward + 1)/2] for step in all_steps]
    
