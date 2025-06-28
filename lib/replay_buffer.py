import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.count = 0  # Total items seen

    def maybe_add(self, item):
        self.count += 1
        if len(self.buffer) < self.max_size:
            # Fill buffer up to max size
            self.buffer.append(item)
        else:
            # Replace existing item with decreasing probability
            idx = random.randint(0, self.count - 1)
            if idx < self.max_size:
                self.buffer[idx] = item

    def get_random(self):
        return random.choice(self.buffer)
