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
        stored_item = [item.board.state, item.reward]
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
            state = game.GameState(board=game.Board(), next_move=1)
            state.board.state = item[0]
            state.reward = item[1]
            return state
        return item

    def to_json(self) -> str:
        return json.dumps(
            {
                "count": self.count,
                "max_size": self.max_size,
                "buffer": self.buffer,
            }
        )

    def from_json(self, jsonx: str) -> None:
        s = json.loads(jsonx)
        self.count = s["count"]
        self.max_size = s["max_size"]
        self.buffer = s["buffer"]
