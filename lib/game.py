import random
import copy
from dataclasses import dataclass
from typing import Optional
from enum import Enum
import math

START_BOARD = [[0 for _ in range(6)] for _ in range(6)]

MCTS_NUM_SIMULATIONS=1000

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
        s = ''
        for row in range(6):
            for col in range(6):
                s += str(self.state[row][col])
        return s

    # Generates all boards for next single step (ply=1 crosses, ply=-1 zeroes)
    # Returns list of tuples. Each tuple is a board and pair of coordinates of the added element
    def all_next_steps(self, ply):
        boards = []
        for row in range(6):
            for col in range(6):
                if self.state[row][col] == 0:
                    next_board = self.copy()  # copy.deepcopy(board)
                    next_board.state[row][col] = ply
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
        b = self.state

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
        b = self.state

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



    def print_board(self, r=None, c=None):

        bgs = {
            #"grey": "\033[100m", 
            #"black": "\033[40m",
            #"yellow": "\033[43m"
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

        winner, xyo = self.check_winner()

        for i, row in enumerate(self.state):
            for j, cell in enumerate(row):

                bg = "grey" if (i + j) % 2 == 0 else "black"
                if cell == -1:
                    what, fg = " O ", "green"
                elif cell == 1:
                    what, fg = " X ", "blue"
                else:
                    what, fg = "   ", "std"
                fg = "red" if (i, j) in xyo else fg
                
                if i == r and j == c:
                   #fg = "yellow"
                   bg = "yellow"

                cprint(fg, bg, what)

            print()


@dataclass
class Step:
    ply: int  # 1 for crosses, -1 for zeroes
    x: int
    y: int
    board: list[list[int]]
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
        
        for board, row, col in boards:
            value = m.get_next_step_value(board.state)
            if value is None:
                continue

            if ply == 1 and value > best:
                best = value
                best_xy = (row, col)
            if ply == -1 and value < best:
                best = value
                best_xy = (row, col)

        return best_xy[0], best_xy[1]

    def best_minimax_step(self, board, ply):
        player = "X" if ply == 1 else "O"
        depth = 2
        alpha, beta = -1000, 1000 # infinity!
        val, row, col = self.minimax(board, depth, alpha, beta, player)
        return row, col


    class MctsNode:
        parent = None # previous step node, None for root
        board: Board
        row: int 
        col: int
        ply: int  ## 1 or -1
        all_moves: list[Board] = []
        tried_nodes: list[Board] = []
        num_visits: int = 0
        value: float = 0.0
        prior: float = 0.0
        is_terminal = False

        def __init__(self, board, row, col, ply):
            self.board = board
            self.row = row
            self.col = col
            self.ply = ply
            self.num_visits = 0
            self.all_moves = self.board.all_next_steps(self.ply)
            self.tried_nodes = []
            if len(self.all_moves) == 0:
                self.is_terminal = True
 

    def mcts_get_best_node_to_continue(self, root):
        tried_nodes = root.tried_nodes
        n_total = root.num_visits
        n_sqrt_total = math.sqrt(n_total)
        c_puct = 1.0 # TODO: experiment with that
        best_puct = -1000 * root.ply # negative infinity for X, positive for O
        best_node = None
        for node in tried_nodes:
            node_puct = node.value / node.num_visits + node.prior * n_sqrt_total / node.num_visits
            if root.ply == 1 and node_puct > best_puct:
                best_puct = node_puct
                best_node = node
            if root.ply == -1 and node_puct < best_puct:
                best_puct = node_puct
                best_node = node
        return best_node
 

    # Currently the values are taken from the model value outputs, so they are not normalized
    # Therefore have to softmax them for puct to work
    # TODO: make a policy network and take those priors from there
    def mcts_softmax_priors(self, tried_nodes):
        priors = [node.value for node in tried_nodes]

        # softmax
        m = max(priors)
        exps = [math.exp(v - m) for v in priors]
        total = sum(exps)
        priors = [e / total for e in exps]

        for i, node in enumerate(tried_nodes):
            node.prior = priors[i]

    # Returns the last visited node
    def mcts_run_simulation(self, root):
        if root.is_terminal:
            # no more walking after terminal
            return root

        tried, all = len(root.tried_nodes), len(root.all_moves)
        if tried < all:
            board, row, col = root.all_moves[tried]

            next_node = Game.MctsNode(board, row, col, -root.ply)
            m = self.model_x if next_node.ply == 1 else self.model_o

            winner, _ = board.check_winner()
            if winner == 1 or winner == -1:
              next_node.value = winner
              next_node.is_terminal = True
            else:
              next_node.value = m.get_next_step_value(board.state)
            next_node.parent = root

            root.tried_nodes.append(next_node)

            if tried == all - 1: # we just added last move, time to calc priors
                self.mcts_softmax_priors(root.tried_nodes)

            return next_node

        next_node = self.mcts_get_best_node_to_continue(root)
        return self.mcts_run_simulation(next_node)


    def mcts_back_propagate(self, node):
        cur_node = node
        while cur_node is not None:
            cur_node.num_visits += 1
            cur_node.value += node.value
            cur_node = cur_node.parent
        return

    
    def mcts_node_count(self, root):
        count = 0
        for node in root.tried_nodes:
            count += self.mcts_node_count(node)
        return count + 1

    def mcts_depth(self, root):
        depth = 0
        if len(root.tried_nodes) == 0:
            return 1

        for node in root.tried_nodes:
            subdepth = self.mcts_depth(node)
            if subdepth > depth:
                depth = subdepth
        return depth + 1

    def mcts_unique_node_count(self, root, accum=None):
        if accum is None:
            accum = {}
        hsh = root.board.asstr()
        accum[hsh] = 1

        for node in root.tried_nodes:
            self.mcts_unique_node_count(node, accum)
        return len(accum.keys())

    def mcts_terminal_node_count(self, root, accum=None):
        if root.is_terminal:
            return 1
        cnt = 0
        for node in root.tried_nodes:
            cnt += self.mcts_terminal_node_count(node)
        return cnt

    def mcts_analyze_tree(self, root):
        print("MCTS available moves: ", len(root.all_moves))
        print("MCTS node count: ", self.mcts_node_count(root))
        print("MCTS unique count: ", self.mcts_unique_node_count(root))
        print("MCTS terminal count: ", self.mcts_terminal_node_count(root))
        print("MCTS tree depth: ", self.mcts_depth(root))
        print()

    def best_mcts_step(self, board, ply):
        root = Game.MctsNode(board, None, None, ply)

        # TODO: make it a param
        num_simulations = MCTS_NUM_SIMULATIONS
        for sim_num in range(num_simulations):
            last_node = self.mcts_run_simulation(root)
            self.mcts_back_propagate(last_node)

        self.mcts_analyze_tree(root)


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
      if winner != 0:
          return (winner, None, None)
      
      m = self.model_x if player == 'X' else self.model_o 
      if depth == 0:
          return m.get_next_step_value(board.state), None, None

      boards = board.all_next_steps(1 if player == "X" else -1) #ply)
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


    def step_no(self):
        # Step number is count of O's on the board.
        return sum([1 for row in self.board.state for x in row if x == -1])


    def choose_next_step(self, ply):
        # First step is always random to increase diversity
        if self.step_no() == 0: 
              return self.random_step()

        if self.game_mode == "minimax":
           return self.best_minimax_step(self.board, ply)

        if self.game_mode == "mcts":
           return self.best_mcts_step(self.board, ply)

        return self.best_greedy_step(self.board, ply)


    def play_game(self):
        self.board.reset()
        steps, ply, winner = [], 1, 0
        while True:
            x, y = self.choose_next_step(ply)
            self.board.state[x][y] = ply

            steps.append(Step(board=self.board.copy(), ply=ply, x=x, y=y))
            ply = -ply

            winner, _ = self.board.check_winner()
            if len(steps) == 36 or winner != 0:
                break

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


def competition(model_x, model_o, num_games, game_type=GameType.TICTACTOE_6_6_4, game_mode="greedy"):
    winners = {-1: 0, 0: 0, 1: 0}
    g = Game(model_x, model_o, game_type, game_mode)
    for f in range(num_games):
        #print(f"Game {f}")
        _, winner = g.play_game()
        winners[winner] = winners[winner] + 1

    return winners
