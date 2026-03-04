import math
import random

def scores(matches: list[tuple[str, int, str, int]]) -> dict[str, int]:
    scores: dict[str, int] = {}
    for p1, w1, p2, w2 in matches:
        scores.setdefault(p1, 0)
        scores.setdefault(p2, 0)
        scores[p1] += w1
        scores[p2] += w2
    return scores
         

def elo_ratings(matches: list[tuple[str, int, str, int]], expand_matches: bool = False) -> dict[str, float]:
    """
    matches: list of (player1, wins1, player2, wins2)
    returns: list of (player, rating)
    """
    ratings: dict[str, float] = {}
    K_per_game=20
    initial_rating=1500

    if expand_matches:
        expanded = []
        for p1, w1, p2, w2 in matches:
            for _ in range(w1):
                expanded.append((p1, 1, p2, 0))
            for _ in range(w2):
                expanded.append((p1, 0, p2, 1))
        matches = expanded
    random.shuffle(matches)

    for p1, w1, p2, w2 in matches:
        r1 = ratings.setdefault(p1, initial_rating)
        r2 = ratings.setdefault(p2, initial_rating)

        total = w1 + w2
        s1 = w1 / total
        s2 = w2 / total

        E1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
        E2 = 1 / (1 + 10 ** ((r1 - r2) / 400)) # = 1 - E1

        K_match = K_per_game * total

        ratings[p1] += K_match * (s1 - E1)
        ratings[p2] += K_match * (s2 - E2)

    return ratings

from typing import Any
def second_best(inputs: dict[str, Any]) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    def outs(figure: str) -> list[tuple[str, float]]:
      return sorted([(name, rating) for name, rating in inputs.items() if figure in name], key=lambda x: -x[1])

    return outs("crosses"), outs("zeroes")

