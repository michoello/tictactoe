from __future__ import annotations

from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
from lib import ttt_player_v2 as tttv2
from lib import pickup_model
from lib import replay_buffer
from lib import ratings

from zoneinfo import ZoneInfo
from utils import run_parallel
from listinvert import value

import math
import sys
import os
import time
import shutil
from typing import Any
import random
import builtins
from datetime import datetime
import re

# -------------------------------------------
# Prepend each line with datetime
prev_ts = -1
_TS_RE = re.compile(r"\[ts:(\d+)\]")

def timestamped_print(*args: Any, sep: str = " ", end: str = "\n", file: Any = None, flush: bool = False) -> int:
    ts = int(time.time())

    global prev_ts
    if prev_ts == -1:
        prev_ts = ts

    timestamp = datetime.now(ZoneInfo('America/Los_Angeles')).strftime("%Y-%m-%d %H:%M:%S")
    message = sep.join(str(arg) for arg in args)
    m = _TS_RE.search(message)
    if m:
        prev_ts = int(m.group(1))
        message = _TS_RE.sub("", message, count=1).strip()

    dif_ts = ts - prev_ts
    if dif_ts > 1:
        prev_ts = ts

    lines = message.splitlines()
    timestamped_lines = [f"{dif_ts:5d} - {timestamp} - {line}" for line in lines]
    final_message = "\n".join(timestamped_lines)
    builtins.print(final_message, end=end, file=file, flush=flush)
    return ts

print = timestamped_print

# Duplicate all output to a file
class Tee:
    def __init__(self, *streams: Any) -> None:
        self.streams = streams
    def write(self, data: Any) -> None:
        for s in self.streams:
            s.write(data)
    def flush(self) -> None:
        for s in self.streams:
            s.flush()

# -------------------------------------------

def calc_loss(player: int, m: Any, boards: list[list[list[int]]], values: list[list[list[float]]]) -> float:
    sum_loss = 0
    for board, state_value in zip(boards, values):
        m.set_board_and_value(player, board, state_value)
        sum_loss = sum_loss + m.get_loss_value()[0]
    return sum_loss / len(boards)

def grad_norm(grads: list[list[float]]) -> float:
  total_sq = 0.0
  for row in grads:
    for g in row:
      total_sq += g * g
  return math.sqrt(total_sq)

def train_single_round(
    trainee: str,
    model_x: Any,
    model_o: Any,
    m_student: Any,
    batch_size: int,
    train_iterations: int
) -> None:
    g = game.Game(model_x, model_o)
    train_boards, train_values = g.generate_batch_from_games(batch_size)

    # Get old memories from buffer
    replay_buffer = m_student.replay_buffer()
    replay_boards, replay_values = [], []
    if replay_buffer.count > 100:
        for i in range(16):
            rr = replay_buffer.get_random()
            replay_boards.append(rr[0])
            replay_values.append(rr[1])

    for i in range(len(train_boards)):
        replay_buffer.maybe_add([train_boards[i], train_values[i]])

    train_boards.extend(replay_boards)
    train_values.extend(replay_values)

    # Backward pass
    for i in range(train_iterations):
        sloss, k1norm, k2norm, cnt = 0.0, 0.0, 0.0, 0
        for board, state_value in zip(train_boards, train_values):
            _player = 1 if trainee == "crosses" else -1
            m_student.set_board_and_value(_player, board, state_value)
            m_student.calc_grads()
            m_student.apply_gradient(0.001)

            loss, policy_loss = m_student.get_loss_value()

            if math.isnan(loss):
              print("Label: ", value(m_student.impl.value_label.fval()))
              print("input: ", value(m_student.impl.dinput.fval()))
              print("Grads kernels1: ", value(m_student.impl.kernels1.bval()))
              print("Grads kernels2: ", value(m_student.impl.kernels2.bval()))
              print("Output kernels1: ", value(m_student.impl.kernels1.fval()))
              print("Output kernels2: ", value(m_student.impl.kernels2.fval()))
              import sys
              sys.exit()

            sloss += loss
            k1norm += grad_norm(value(m_student.impl.kernels1.bval()))
            k2norm += grad_norm(value(m_student.impl.kernels2.bval()))
            cnt += 1

        train_loss = calc_loss(_player, m_student, train_boards, train_values)
        if i % 20 == 0:
            sloss, k1norm, k2norm = sloss/cnt, k1norm/cnt, k2norm/cnt
            print(f"EPOCH {i}: Train loss={train_loss}")
            print("    Loss: ", sloss, " k1norm=", k1norm, "  k2norm=", k2norm)

def model_name(prefix: str, family: Any, trainee: str, version: int) -> str:
    if family is None:
        return f"{prefix}-{trainee}-{version}.json"
    return f"{prefix}-{trainee}-{family}.{version}.json"

def model_name_duo(prefix: str, version: int) -> str:
    return f"{prefix}-{version}.json"

def sorted_sample(n: int, m: int) -> list[int]:
    return list(range(n)) if n <= m else sorted(random.sample(range(n), m))

NUM_GAMES = 20

def fight(trainee: str, student_path: str, opponent_type: str, opponent_path: str) -> Any:
    m_student = tttv2.TTTPlayerV2(student_path)
    m_opponent = pickup_model(opponent_type, opponent_path)

    if trainee == "crosses":
        model_x, model_o = m_student, m_opponent
    else:
        model_x, model_o = m_opponent, m_student

    g = game.Game(model_x, model_o)
    winners = g.competition(NUM_GAMES)
    print(f"FIGHT! {student_path} vs {opponent_path}: {winners}")

    return winners

def versioned_competition(
    prefix: str,
    family: str,
    version: int,
    trainee: str,
    opponents: list[Any]
) -> None:
    opponent = "crosses" if trainee == "zeroes" else "zeroes"
    student_model = model_name(prefix, family, trainee, version)

    tasks = []
    for opponent_model in opponents:
        tasks.append((fight, [trainee, student_model, *opponent_model]))

    all_winners = run_parallel(tasks, max_workers=1)

    losing_versions: list[Any] = []
    total, winning = 0, 0
    for winners in all_winners:
        total += winners[-1] + winners[1]
        if trainee == "zeroes":
            winning += winners[-1]
        if trainee == "crosses":
            winning += winners[1]

    win_ratio = round(winning / total, 3)
    print(
        f"Competitions of {student_model} completed: victory ratio is {win_ratio} ({winning} out of {total})"
    )

def clone_new_version(prefix: str, family: str, from_version: int, to_version: int) -> None:
    shutil.copyfile(
        model_name(prefix, family, "crosses", from_version),
        model_name(prefix, family, "crosses", to_version),
    )
    shutil.copyfile(
        model_name(prefix, family, "zeroes", from_version),
        model_name(prefix, family, "zeroes", to_version),
    )

def train(
    prefix: str,
    family_cross: str,
    family_zero: str,
    version: int,
    trainee: str,
    num_rounds: int,
    batch_size: int,
    train_iterations: int
) -> None:
    crosses_name = model_name(prefix, family_cross, "crosses", version)
    zeroes_name = model_name(prefix, family_zero, "zeroes", version)
    model_x = tttv2.TTTPlayerV2(crosses_name)
    model_o = tttv2.TTTPlayerV2(zeroes_name)

    if trainee == "crosses":
        m_student, m_opponent = model_x, model_o
        student_name, opponent_name = crosses_name, zeroes_name
        student_family = family_cross
    else:
        m_student, m_opponent = model_o, model_x
        student_name, opponent_name = zeroes_name, crosses_name
        student_family = family_zero

    print("-------------------------------------------------")
    tr_ts = print(f"Start {student_name}")
    for i in range(num_rounds):
        it_ts = print(f"Start {student_name} vs {opponent_name} ITER {i}")
        train_single_round(trainee, model_x, model_o, m_student, batch_size, train_iterations)
        print(
            f"[ts:{it_ts}] Finish {student_name} vs {opponent_name} ITER {i}"
        )

    old_version = version - 5
    old_student_name = model_name(prefix, student_family, trainee, old_version)
    if old_version > 2 and old_version % 100 != 0:
       print(f"Deleting {old_student_name}")
       try:
           os.remove(old_student_name)
       except FileNotFoundError:
           pass

    student_name = model_name(prefix, student_family, trainee, version + 1)
    m_student.save_to_file(student_name)
    print(f"[ts:{tr_ts}] Saved {student_name}")

def cross_competition(prefix: str, families: list[str], version: int) -> None:
    start_ts = print(f"Cross competition {version} started")
    matches = []
    for i in range(len(families)):
        for j in range(len(families)):
            crosses_path = model_name(prefix, families[i], "crosses", version)
            zeroes_path = model_name(prefix, families[j], "zeroes", version)
            model_x = tttv2.TTTPlayerV2(crosses_path)
            model_o = tttv2.TTTPlayerV2(zeroes_path)
            g = game.Game(model_x, model_o)
            winners = g.competition(NUM_GAMES)
            print(f"TGIFH: {crosses_path} vs {zeroes_path}: {winners}")
            matches.append((crosses_path, winners[1], zeroes_path, winners[-1]))
     
    print("CrossCompete results")
    results = ratings.second_best(ratings.scores(matches))
    for cross, zero in zip(*results):
       print(f"X: {cross} \t O: {zero}")

    print(f"[ts:{start_ts}] Cross competition {version} finished")

def main_loop(
    prefix: str,
    families: list[str],
    max_workers: int,
    max_version: int,
    num_rounds: int,
    batch_size: int,
    train_iterations: int,
    opponents: list[Any]
) -> None:
    version = 0
    for family in families:
       model_x = tttv2.TTTPlayerV2()
       model_x.save_to_file(model_name(prefix, family, "crosses", version))
       model_o = tttv2.TTTPlayerV2()
       model_o.save_to_file(model_name(prefix, family, "zeroes", version))

    while version < max_version:
        cross_families = random.sample(families, len(families))
        zero_families = random.sample(families, len(families))

        start_ts = print(f"Training for version {version} started")
        tasks = []
        for family_cross, family_zero in zip(cross_families, zero_families):
            tasks.append((train, [prefix, family_cross, family_zero, version, "crosses", num_rounds, batch_size, train_iterations]))
            tasks.append((train, [prefix, family_cross, family_zero, version, "zeroes", num_rounds, batch_size, train_iterations]))
        results = run_parallel(tasks, max_workers=max_workers)
        print(f"[ts:{start_ts}] Training for version {version} finished")

        version += 1

        if version % 1 == 0:
            start_ts = print(f"Competitions for version {version} started")
            for family in families:
              versioned_competition(prefix, family, version, "crosses", opponents)
              versioned_competition(prefix, family, version, "zeroes", opponents)
            print(f"[ts:{start_ts}] Competition for version {version} finished")

            cross_competition(prefix, families, version)
