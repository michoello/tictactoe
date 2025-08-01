import random

class ReplayBuffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = []
        self.count = 0  # Total items seen

    # Returns True if item is added
    def maybe_add(self, item):
        self.count += 1
        added = False
        if len(self.buffer) < self.max_size:
            # Fill buffer up to max size
            self.buffer.append(item)
            return True
        else:
            # Replace existing item with decreasing probability
            idx = random.randint(0, self.count - 1)
            if idx < self.max_size:
                self.buffer[idx] = item
                return True
        return False

    def get_random(self):
        return random.choice(self.buffer)
