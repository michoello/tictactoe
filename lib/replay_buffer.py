import random
import json
from typing import Any
from .game import GameState


class ReplayBuffer:
    max_size: int
    buffer: list[Any]
    count: int

    def __init__(self, max_size: int) -> None:
        self.max_size = max_size
        self.buffer = []
        self.count = 0  # Total items seen

    # Returns True if item is added
    def maybe_add(self, item: GameState) -> bool:
        stored_item = [item.board.cells, item.reward]
        self.count += 1
        added = False
        if len(self.buffer) < self.max_size:
            # Fill buffer up to max size
            self.buffer.append(stored_item)
            return True
        else:
            # Replace existing item with decreasing probability
            idx = random.randint(0, self.count - 1)
            if idx < self.max_size:
                self.buffer[idx] = stored_item
                return True
        return False

    def get_random(self) -> GameState:
        item = random.choice(self.buffer)
        if isinstance(item, list) and len(item) == 2:
            import lib.game as game
            state = game.GameState(board=game.Board(), next_player=1)
            
            # handle board payload
            if isinstance(item[0], game.Matrix):
                state.board.cells = game.Matrix(item[0])
            else:
                m = game.Matrix(6, 6)
                m.set_data(item[0])
                state.board.cells = m

            # handle reward payload
            if item[1] is None:
                state.reward = None
            elif isinstance(item[1], game.Matrix):
                state.reward = game.Matrix(item[1])
            else:
                r = game.Matrix(1, 1)
                r.set(0, 0, item[1][0][0])
                state.reward = r

            return state
        return item

    def to_json(self) -> str:
        from listinvert import value as mx_value
        import lib.game as game
        
        serializable_buffer = []
        for m, r in self.buffer:
            m_val = mx_value(m) if isinstance(m, game.Matrix) else m
            r_val = mx_value(r) if isinstance(r, game.Matrix) else r
            serializable_buffer.append([m_val, r_val])
            
        return json.dumps(
            {
                "buffer": serializable_buffer,
                "count": self.count,
                "max_size": self.max_size,
            }
        )

    def from_json(self, jsonx: str) -> None:
        s = json.loads(jsonx)
        self.count = s["count"]
        self.max_size = s["max_size"]
        self.buffer = s["buffer"]
