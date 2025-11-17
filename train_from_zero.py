from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
from lib import pickup_model
from lib import replay_buffer

from utils import run_parallel

import sys
import os
import time
import shutil
from typing import Any
import random
import argparse


from multiprocessing import Process
import builtins
from datetime import datetime
import re

# -------------------------------------------
# Prepend each line with datetime
prev_ts = -1
_TS_RE = re.compile(r"\[ts:(\d+)\]")


def timestamped_print(*args, sep=" ", end="\n", file=None, flush=False):
    ts = int(time.time())

    global prev_ts
    if prev_ts == -1:
        prev_ts = ts

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


logfile = open("output.log", "w")  # TODO: cli arg?
sys.stdout = Tee(sys.stdout, logfile)
# -------------------------------------------


parser = argparse.ArgumentParser(description="Train your model")
parser.add_argument("--init_model", type=str, help="Path to the initial model file")
parser.add_argument("--save_to_model", type=str, help="Path to save the trained model")

args = parser.parse_args()

print(f"Init model: {args.init_model}")
print("Save to model:", args.save_to_model)


def generate_playing_batch(m_crosses, m_zeroes, m_student):
    boards, values = [], []

    g = game.Game(m_crosses, m_zeroes)
    steps, winner = g.play_game(0.3, 5)  ## 0.5, 2
    for step in steps:
        # include only 10% of boards
        if random.random() > 0.9:
            boards.append(step.board.board)
            train_reward = [(step.reward + 1) / 2]
            values.append(train_reward)
    return boards, values


def generate_dumb_batch(num_boards, m_crosses, m_zeroes, m_student):
    outboards, outvalues = [], []
    num_games_played = 0

    i = 0
    while i < num_boards:
        num_games_played += 1
        boards, values = generate_playing_batch(m_crosses, m_zeroes, m_student)
        for board, value in zip(boards, values):
            outboards.append(board)
            outvalues.append(value)
            i += 1

    # Shuffle the batch
    combined = list(zip(outboards, outvalues))
    random.shuffle(combined)
    boards_shuffled, values_shuffled = zip(*combined)
    outboards, outvalues = list(boards_shuffled), list(values_shuffled)

    print(
        f"DUMB BATCH READY: num_boards={len(outboards)} games_played={num_games_played}"
    )
    return outboards, outvalues


def calc_loss(m, boards, values):
    sum_loss = 0
    for board, value in zip(boards, values):
        #m.x.set(board)
        #m.y.set([value])
        m.m.set_data(m.x, board)
        m.m.set_data(m.y, [value])

        m.loss.calc_fval()
        loss = m.loss.fval()
        sum_loss = sum_loss + loss[0][0]

    return sum_loss / len(boards)


BATCH_SIZE = 32
TRAIN_ITERATIONS = 25


def train_single_round(trainee, m_crosses, m_zeroes, m_student):

    train_boards, train_values = generate_dumb_batch(
        BATCH_SIZE, m_crosses, m_zeroes, m_student
    )

    #
    # Get old memories from buffer
    #
    replay_buffer = m_student.replay_buffer
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

    #
    # Backward pass
    #
    for i in range(TRAIN_ITERATIONS):
        for board, value in zip(train_boards, train_values):
            m_student.m.set_data(m_student.x, board)
            m_student.m.set_data(m_student.y, [value])

            m_student.loss.calc_fval()
            m_student.calc_grads()
            m_student.apply_gradient(0.01)

            m_student.loss.calc_fval()
            loss = m_student.loss.fval()

        train_loss = calc_loss(m_student, train_boards, train_values)
        if i % 10 == 0:
            print(f"EPOCH {i}: Train loss={train_loss}")


def model_name(prefix, family, trainee, version):
    if family is None:
        return f"{prefix}-{trainee}-{version}.json"
    return f"{prefix}-{trainee}-{family}.{version}.json"


def sorted_sample(n, m):
    return list(range(n)) if n <= m else sorted(random.sample(range(n), m))


NUM_GAMES = 10


def fight(trainee, student_model, opponent_type, opponent_path):
    m_student = tttp.TTTPlayer(student_model)
    m_opponent = pickup_model(opponent_type, opponent_path)

    if trainee == "crosses":
        m_crosses, m_zeroes = m_student, m_opponent
    else:
        m_crosses, m_zeroes = m_opponent, m_student

    winners = game.competition(m_crosses, m_zeroes, NUM_GAMES)
    print(f"FIGHT! {student_model} vs {opponent_path}: {winners}")

    return winners


# Returns true if student wins over previous version
def versioned_competition(prefix, family, version, trainee):
    opponent = "crosses" if trainee == "zeroes" else "zeroes"

    student_model = model_name(prefix, family, trainee, version)

    # Collect the opponents to play against
    opponents = []
    # get sample of past versions, and prev one for sure
    # for v in sorted_sample(version-1, 4) + [version-1]:
    #    opponents.append(["player", model_name(prefix, opponent, v)])

    # play with classifier - our first baseline
    # opponents.append(["classifier", "models/model_victory_only.json"])

    # play against "replay buffer version", first stable
    # prv_prefix = "models/with_replay_buffer/model"
    # prv_prefix = "models/fixed_rounds/model"
    #prv_prefix = "models/shorter_rounds/model"
    #for prv_v in [1000, 1100, 1132]:
    #    opponents.append(["player", model_name(prv_prefix, None, opponent, prv_v)])
    prv_prefix = "models/try_again/model"
    for prv_v in [1440, 2400]:
        opponents.append(["player", model_name(prv_prefix, None, opponent, prv_v)])
    prv_prefix = "models/cpp/model"
    for prv_v in [1000, 3000, 13000]:
        opponents.append(["player", model_name(prv_prefix, None, opponent, prv_v)])

    # This parallelization does not help yet, but let's keep for future improvements
    tasks = []
    for opponent_model in opponents:
        tasks.append((fight, [trainee, student_model, *opponent_model]))

    all_winners = run_parallel(tasks, max_workers=2)

    losing_versions = []
    total, winning = 0, 0
    for winners in all_winners:
        total += winners[-1] + winners[1]
        if trainee == "zeroes":  # and winners[-1] < winners[1]:
            winning += winners[-1]
        if trainee == "crosses":  # and winners[1] < winners[-1]:
            winning += winners[1]

    win_ratio = round(winning / total, 3)
    print(
        f"Competitions of {student_model} completed: victory ratio is {win_ratio} ({winning} out of {total})"
    )


def clone_new_version(prefix, family, from_version, to_version):
    shutil.copyfile(
        model_name(prefix, family, "crosses", from_version),
        model_name(prefix, family, "crosses", to_version),
    )
    shutil.copyfile(
        model_name(prefix, family, "zeroes", from_version),
        model_name(prefix, family, "zeroes", to_version),
    )


# --------------------------------------------
NUM_ROUNDS = 2


def train(prefix, family_cross, family_zero, version, trainee):
    crosses_name = model_name(prefix, family_cross, "crosses", version)
    zeroes_name = model_name(prefix, family_zero, "zeroes", version)
    m_crosses = tttp.TTTPlayer(crosses_name)
    m_zeroes = tttp.TTTPlayer(zeroes_name)

    if trainee == "crosses":
        m_student, m_opponent = m_crosses, m_zeroes
        student_name, opponent_name = crosses_name, zeroes_name
        student_family = family_cross
    else:
        m_student, m_opponent = m_zeroes, m_crosses
        student_name, opponent_name = zeroes_name, crosses_name
        student_family = family_zero

    print("-------------------------------------------------")
    tr_ts = print(f"Start {student_name}\n\n")
    for i in range(NUM_ROUNDS):
        it_ts = print(f"Start {student_name} vs {opponent_name} ITER {i}")
        train_single_round(trainee, m_crosses, m_zeroes, m_student)
        print(
            f"[ts:{it_ts}] Finish {student_name} vs {opponent_name} ITER {i}"
        )

    # Delete old files to prevent disk overflow
    old_version = version - 5
    old_student_name = model_name(prefix, student_family, trainee, old_version)
    if old_version > 2 and old_version % 100 != 0:
       print(f"Deleting {old_student_name}")
       os.remove(old_student_name)
    else:
       print(f"Not Deleting {old_student_name}")

    # Creating next version
    student_name = model_name(prefix, student_family, trainee, version + 1)
    m_student.save_to_file(student_name)
    print(f"[ts:{tr_ts}] Saved {student_name}")


def main():
    prefix = args.save_to_model

    families = ['a', 'b', 'c', 'd']
    version = 0
    for family in families:
       m_crosses = tttp.TTTPlayer(enable_cpp=True)
       m_crosses.save_to_file(model_name(prefix, family, "crosses", version))
       m_zeroes = tttp.TTTPlayer(enable_cpp=True)
       m_zeroes.save_to_file(model_name(prefix, family, "zeroes", version))


    while True:
        #
        # Train
        #
        cross_families = random.sample(families, len(families))
        zero_families = random.sample(families, len(families))

        start_ts = print(f"Training for version {version} started")
        tasks = []
        for family_cross, family_zero in zip(cross_families, zero_families):
            tasks.append((train, [prefix, family_cross, family_zero, version, "crosses"]))
            tasks.append((train, [prefix, family_cross, family_zero, version, "zeroes"]))
        results = run_parallel(tasks, max_workers=2)
        print(f"[ts:{start_ts}] Training for version {version} finished")

        # Next
        version += 1

        #
        # Compete and check if student wins now - this is optional and unnecessary here
        # TODO: extract into a separate tool
        #
        if version % 100 == 0:
            start_ts = print(f"Competitions for version {version} started")
            for family in families:
              versioned_competition(prefix, family, version, "crosses")
              versioned_competition(prefix, family, version, "zeroes")
            print(f"[ts:{start_ts}] Competition for version {version} finished")

        if version == 4001:
           sys.exit(0)


if __name__ == "__main__":
    main()
