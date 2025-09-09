from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
from lib import replay_buffer
import sys
import os
import time
import shutil
from typing import Any
import random
import argparse

from utils import run_parallel

from multiprocessing import Process
import builtins
from datetime import datetime
import re

# -------------------------------------------
# Prepend each line with datetime
prev_ts = -1
_TS_RE = re.compile(r"\[ts:(\d+)\]")

def timestamped_print(*args, sep=' ', end='\n', file=None, flush=False):
    ts = int(time.time())

    global prev_ts
    if prev_ts == -1:
       prev_ts = ts

    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
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
    final_message = '\n'.join(timestamped_lines)
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

logfile = open("output.log", "w")   # TODO: cli arg?
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

  print( f"DUMB BATCH READY: num_boards={len(outboards)} games_played={num_games_played}")
  return outboards, outvalues



def calc_loss(m, boards, values):
    sum_loss = 0
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value])

        loss = m.loss.val()
        sum_loss = sum_loss + loss[0][0]

    return sum_loss / len(boards)


def train_single_round(trainee, m_crosses, m_zeroes, m_student):

    train_boards, train_values = generate_dumb_batch(32, m_crosses, m_zeroes, m_student)

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
    #for i in range(len(replay_boards)):
    #    train_boards.append(replay_boards[i])
    #    train_values.append(replay_values[i])

    #
    # Backward pass
    #
    train_iterations = 25
    for i in range(train_iterations):
        train_loss = 0
        for board, value in zip(train_boards, train_values):
            m_student.x.set(board)
            m_student.y.set([value])

            m_student.loss.dif()
            m_student.apply_gradient()

            loss = m_student.loss.val()
            train_loss = train_loss + loss[0][0]

        train_loss = calc_loss(m_student, train_boards, train_values)
        print(f"EPOCH {i}: Train loss={train_loss}")


def model_name(prefix, trainee, version):
   return f"{prefix}-{trainee}-{version}.json"


def sorted_sample(n, m):
    return list(range(n)) if n <= m else sorted(random.sample(range(n), m))


# Returns true if student wins over previous version
def versioned_competition(prefix, version, trainee):
    opponent = "crosses" if trainee == "zeroes" else "zeroes"

    student_model = model_name(prefix, trainee, version)
    m_student = tttp.TTTPlayer(student_model)

    #
    # Play against previous versions
    #
    losing_versions = []
    total_games = 0
    for v in sorted_sample(version-1, 4) + [version-1]:  # get sample of past versions, and prev one for sure
        opponent_model = model_name(prefix, opponent, v)
        m_opponent = tttp.TTTPlayer(opponent_model)

        if trainee == "crosses":
           winners = game.competition(m_student, m_opponent, 20)
        else:
           winners = game.competition(m_opponent, m_student, 20)
        print(f"VERSIONED COMPETITION {trainee} VS {opponent_model}: ", winners)

        if trainee == "zeroes" and winners[-1] < 12:
           losing_versions.append(v)
        if trainee == "crosses" and winners[1] < 12:
           losing_versions.append(v)

        total_games += 1

    win_ratio = 1 - len(losing_versions)/total_games

    won_over_prev = version not in losing_versions
    if won_over_prev:
        print(f"VICTORY! STUDENT {trainee}.v{version} WON over {opponent_model}! VICTORY RATIO {win_ratio}")

    #
    # Play against classifier
    #
    opponent_model = "models/model_victory_only.json"
    m_opponent = tttc.TTTClass(opponent_model)
    if trainee == "crosses":
        winners = game.competition(m_student, m_opponent, 20)
    else:
        winners = game.competition(m_opponent, m_student, 20)
    print(f"VICTORY CLASSIFIER COMPETITION {trainee} v{version} won_prev:{won_over_prev}: {winners}, ratio {win_ratio}")

    # 
    # Play against previous best version
    #
    prv_prefix = "models/with_replay_buffer/model"
    #for prv_v in [version, 200, 400, 900]:
    for prv_v in [200, 400, 900]:
        opponent_model = model_name(prv_prefix, opponent, prv_v)
        m_opponent = tttp.TTTPlayer(opponent_model)
        if trainee == "crosses":
            winners = game.competition(m_student, m_opponent, 20)
        else:
            winners = game.competition(m_opponent, m_student, 20)
        print(f"PREVIOUS_BEST MODEL COMPETITION {trainee} v{version} VS {opponent_model}: ", winners)

    return losing_versions


def clone_new_version(prefix, from_version, to_version):
    shutil.copyfile(model_name(prefix, "crosses", from_version), model_name(prefix, "crosses", to_version))
    shutil.copyfile(model_name(prefix, "zeroes", from_version), model_name(prefix, "zeroes", to_version))

# --------------------------------------------
def train(prefix, version, trainee):
    m_crosses = tttp.TTTPlayer(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer(model_name(prefix, "zeroes", version))

    m_student = m_crosses if trainee == "crosses" else m_zeroes
    m_opponent = m_zeroes if trainee == "crosses" else m_crosses

    print("-------------------------------------------------")
    tr_ts = print(f"Start {m_student.file_name}\n\n")
    for i in range(10):
        it_ts = print(f"Start {m_student.file_name} vs {m_opponent.file_name} ITER {i}")
        train_single_round(trainee, m_crosses, m_zeroes, m_student)
        print(f"[ts:{it_ts}] Finish {m_student.file_name} vs {m_opponent.file_name} ITER {i}")

    student_name = model_name(prefix, trainee, version+1)
    m_student.save_to_file(student_name)
    print(f"[ts:{tr_ts}] Saved {student_name}")


def main():
    prefix = args.save_to_model
    
    version = 0
    m_crosses = tttp.TTTPlayer()
    m_crosses.save_to_file(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer()
    m_zeroes.save_to_file(model_name(prefix, "zeroes", version))
    
    while True:
        #
        # Train
        #
        start_ts = print(f"Training for version {version} started")

        tasks = [
          (train, [prefix, version, "crosses"]),
          (train, [prefix, version, "zeroes"]),
        ]

        results = run_parallel(tasks, max_workers=2)


        print(f"[ts:{start_ts}] Training for version {version} finished")


        # Next
        version += 1

        #
        # Compete and check if student wins now - this is optional and unnecessary here
        # TODO: extract into a separate tool
        #
        start_ts = print(f"Competition for version {version} started")

        trainee = "crosses"
        losing_versions = versioned_competition(prefix, version, trainee)
        print(f"COMPETITION {trainee} version {version}: ", "LOSER" if version in losing_versions else "WINNER")

        trainee = "zeroes"
        losing_versions = versioned_competition(prefix, version, trainee)
        print(f"COMPETITION {trainee} version {version}: ", "LOSER" if version in losing_versions else "WINNER")

        print(f"[ts:{start_ts}] Competition for version {version} finished")

        sys.exit(0)



if __name__ == "__main__":
    main()

