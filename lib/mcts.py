from __future__ import annotations
import math
import random
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from lib.game import Board, GameState, Game

# Entire MCTS algorithm is in this class
class MctsNode:
    state: GameState # TODO
    
    # these guys will be replaced by state
    board: Board
    row: Optional[int]
    col: Optional[int]
    next_move: int  ## 1 or -1
    is_terminal: bool = False
    xyo: Optional[list[tuple[int, int]]] = None

    # Possibly too
    state_value: float = 0  # state value taken from model
    prior: float = 0.0

    parent: Optional[MctsNode] = None  # previous step node, None for root

    all_moves: list[tuple[Board, int, int]] = []
    tried_nodes: list[MctsNode] = []

    # These two are to be used to sort out the best move
    value: float = 0.0  # accumulated value collected from child nodes
    num_visits: int = 0  # number of times simulation passed through the node

    def __init__(self, board: Board, row: Optional[int], col: Optional[int], next_move: int) -> None:
        self.board = board
        self.row = row
        self.col = col
        self.next_move = next_move
        self.num_visits = 0
        self.all_moves = self.board.all_next_steps(self.next_move)
        self.tried_nodes = []

    def mcts_get_best_node_to_continue(self) -> MctsNode:
        n_total = self.num_visits
        n_sqrt_total = math.sqrt(n_total)
        c_puct = 1.0  # TODO: experiment with that
        best_puct = -1000.0
        best_node: Optional[MctsNode] = None
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

        assert best_node is not None
        return best_node

    # Currently the values are taken from the model value outputs, so they are not normalized
    # Therefore have to softmax them for puct to work
    # TODO: make a policy network and take those priors from there
    def mcts_softmax_priors(self) -> None:
        next_move = self.next_move
        tried_nodes = self.tried_nodes

        # converting state values to [0;1] range, and inverting them to 1-v for Os
        # It fakes the policy output, by comparing values
        # of all next states, and then converting it to probs
        # TODO: all these ugly tricks should go away
        priors = [node.state_value for node in tried_nodes]
        priors = [(v + 1) / 2.0 for v in priors]
        if next_move == -1:
            priors = [1 - v for v in priors]

        # softmax
        m = max(priors)
        exps = [math.exp(v - m) for v in priors]
        total = sum(exps)
        priors = [e / total for e in exps]

        for i, node in enumerate(tried_nodes):
            node.prior = priors[i]

    # Returns the last visited node
    def mcts_run_simulation(self, gm: Game) -> MctsNode:
        if self.is_terminal:
            # no more walking after terminal
            return self

        tried, all = len(self.tried_nodes), len(self.all_moves)
        if tried < all:
            board, row, col = self.all_moves[tried]

            next_node = MctsNode(board, row, col, -self.next_move)

            winner, xyo = board.check_winner()
            if winner is not None:
                next_node.state_value = winner
                next_node.is_terminal = True
                next_node.xyo = xyo
            else:
                m = gm.model_x if next_node.next_move == 1 else gm.model_o
                next_node.state_value = m.get_next_step_value(next_node.next_move, board.state)

                # Collecting all values is too slow. TODO: policy output
                # brds = [(board.state, r, c) for board, r, c in self.all_moves]
            next_node.parent = self

            self.tried_nodes.append(next_node)

            if tried == all - 1:  # we just added last move, time to calc priors
                self.mcts_softmax_priors()

            return next_node

        next_node = self.mcts_get_best_node_to_continue()
        return next_node.mcts_run_simulation(gm)

    def mcts_back_propagate(self) -> None:
        cur_node: Optional[MctsNode] = self
        while cur_node is not None:
            cur_node.num_visits += 1
            cur_node.value += self.state_value
            cur_node = cur_node.parent

    def mcts_node_count(self) -> int:
        count = 0
        for node in self.tried_nodes:
            count += node.mcts_node_count()
        return count + 1

    def mcts_depth(self) -> int:
        depth = 0
        if len(self.tried_nodes) == 0:
            return 1

        for node in self.tried_nodes:
            subdepth = node.mcts_depth()
            if subdepth > depth:
                depth = subdepth
        return depth + 1

    def mcts_unique_node_count(self, accum: Optional[dict[str, int]] = None) -> int:
        if accum is None:
            accum = {}
        hsh = self.board.asstr()
        accum[hsh] = 1

        for node in self.tried_nodes:
            node.mcts_unique_node_count(accum)
        return len(accum.keys())

    def mcts_terminal_node_count(self) -> int:
        if self.is_terminal:
            return 1
        cnt = 0
        for node in self.tried_nodes:
            cnt += node.mcts_terminal_node_count()
        return cnt

    def mcts_analyze_tree(self) -> None:
        print("MCTS available moves: ", len(self.all_moves))
        print("MCTS node count: ", self.mcts_node_count())
        print("MCTS unique count: ", self.mcts_unique_node_count())
        print("MCTS terminal count: ", self.mcts_terminal_node_count())
        print("MCTS tree depth: ", self.mcts_depth())
        print()


def best_mcts_step(gm: Game, prev_state: GameState, num_simulations: int) -> GameState:
    board = prev_state.board.copy()
    next_move = prev_state.next_move

    root = MctsNode(board, None, None, next_move)

    for sim_num in range(num_simulations):
        new_leaf_node = root.mcts_run_simulation(gm)
        new_leaf_node.mcts_back_propagate()

    root.mcts_analyze_tree()

    # Choose the node that got the most visits
    best_node: Optional[MctsNode] = None
    best_count = -1
    for node in root.tried_nodes:
        # TODO: break the even: if counts are the same, compare values
        if node.num_visits > best_count:
            best_count = node.num_visits
            best_node = node
    
    assert best_node is not None
    assert best_node.row is not None and best_node.col is not None
    
    board.state[best_node.row][best_node.col] = next_move
    winner = None
    xyo = None
    if best_node.is_terminal:
        winner=int(best_node.state_value)
        xyo=best_node.xyo
    else:
        # verify winner
        winner, xyo = board.check_winner()

    return type(prev_state)(
        board=board,
        next_move=-next_move,
        step_no=prev_state.step_no + 1,
        x=best_node.row,
        y=best_node.col,
        winner=winner,
        xyo=xyo
    )
