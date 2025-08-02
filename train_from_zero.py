from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
from lib import replay_buffer
import sys
import os
from typing import Any
import random


import argparse

parser = argparse.ArgumentParser(description="Train your model")

parser.add_argument("--init_model", type=str, help="Path to the initial model file")
parser.add_argument("--save_to_model", type=str, help="Path to save the trained model")


# -------------------------------------------
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

# Open your log file
logfile = open("output.log", "w")

# Redirect stdout
sys.stdout = Tee(sys.stdout, logfile)
# -------------------------------------------

args = parser.parse_args()

print(f"Init model: {args.init_model}")
print("Save to model:", args.save_to_model)

def generate_playing_batch(num_games, m_crosses, m_zeroes, trainee):
    boards, values = [], []

    g = game.Game(m_crosses, m_zeroes)

    i = 0
    while True:
        #steps, winner = g.play_game(0.5, 2)
        steps, winner = g.play_game(0.3, 5)
        if trainee == "zeroes" and winner == -1:
            i += 1
            #continue
        if trainee == "crosses" and winner == 1:
            i += 1
            #continue
        for step in steps:
            # include only 10% of boards
            if random.random() > 0.9:
               boards.append(step.board.board)
               train_reward = [(step.reward + 1) / 2]
               values.append(train_reward)
        break
    #print(f"WHILE GENERATING BATCH FOR TRAINING, THE NUMBER OF GAMES {trainee} WON IS {i}")
    return boards, values


def generate_dumb_batch(num_boards, trainee, versions_to_train_on, prefix):
  #version = versions_to_train_on[-1]

  outboards, outvalues = [], []
  num_games_played = 0

  for version in versions_to_train_on:

    m_crosses = tttp.TTTPlayer(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer(model_name(prefix, "zeroes", version))

    i = 0
    while i < num_boards: 
        num_games_played += 1
        boards, values = generate_playing_batch(1, m_crosses, m_zeroes, trainee)
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
    "BALANCED BATCH READY: num_boards=",
    len(outboards),
    "games_played=",
    num_games_played,
  )
  return outboards, outvalues



def calc_loss(m, boards, values):
    sum_loss = 0
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value])

        loss = m.loss.val()
        sum_loss = sum_loss + loss[0][0]

    return sum_loss / len(boards)

replay_buffers = {}
replay_buffers["crosses"] = replay_buffer.ReplayBuffer(2000)
replay_buffers["zeroes"] = replay_buffer.ReplayBuffer(2000)


def train_single_epoch(epoch, prefix, version, trainee, versions_to_train_on):
    if len(versions_to_train_on) == 0:
       versions_to_train_on.append(version)

    print("-------------------------------------------------")
    print(f"TRAINING {trainee} {version} EPOCH {epoch} - VERSIONS TO TRAIN ON {versions_to_train_on}")

    train_boards, train_values = generate_dumb_batch(32, trainee, versions_to_train_on, prefix)

    m_student = tttp.TTTPlayer(model_name(prefix, trainee, version))

    #replay_buffer = m_student.replay_buffer ## TODO
    replay_buffer = replay_buffers[trainee]
    replay_boards, replay_values = [], []
    print("STUDENT REPLAY BUFFER COUNT: ", replay_buffer.count)
    if replay_buffer.count > 100:
       for i in range(16):
          rr = replay_buffer.get_random()
          replay_boards.append(rr[0])
          replay_values.append(rr[1])

    replay_added = 0
    for i in range(len(train_boards)):
        if replay_buffer.maybe_add([train_boards[i], train_values[i]]):
           replay_added += 1
    print("MEMORIZED (ADDED TO BUFFER): ", replay_added, "BOARDS")

    print("OLD MEMORIES TOBE USED", len(replay_boards))
    for i in range(len(replay_boards)):
        train_boards.append(replay_boards[i])
        train_values.append(replay_values[i])

    # Backward pass
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
        print(f"EPOCH {epoch}/{i}: Train loss={train_loss}")

    student_model = model_name(prefix, trainee, version)
    m_student.save_to_file(student_model) 
    print(f"SAVED {student_model}")

def model_name(prefix, trainee, version):
   return f"{prefix}-{trainee}-{version}.json"


# Returns true if student wins over previous version
# TODO: check ALL prev versions victory
def versioned_competition(trainee, version, prefix):

    opponent = "crosses" if trainee == "zeroes" else "zeroes"

    student_model = model_name(prefix, trainee, version)
    m_student = tttp.TTTPlayer(student_model)

    #
    # Play against previous versions
    #
    losing_versions = []
    for v in range(0, version + 1):
        opponent_model = model_name(prefix, opponent, v)
        m_opponent = tttp.TTTPlayer(opponent_model)

        if trainee == "crosses":
           winners = game.competition(m_student, m_opponent, 20)
        else:
           winners = game.competition(m_opponent, m_student, 20)
        print(f"VERSIONED COMPETITION RESULTS {trainee} VS {opponent_model}: ", winners)

        if trainee == "zeroes" and winners[-1] < 12:
           losing_versions.append(v)
           
        
        if trainee == "crosses" and winners[1] < 12:
           losing_versions.append(v)

    won_over_prev = version not in losing_versions
    if won_over_prev:
        print(f"VICTORY! STUDENT {trainee}.v{version} WON over {opponent_model}!!!")

    #
    # Play against classifier
    #
    opponent_model = "models/model_victory_only.json"
    m_opponent = tttc.TTTClass(opponent_model)
    if trainee == "crosses":
        winners = game.competition(m_student, m_opponent, 20)
    else:
        winners = game.competition(m_opponent, m_student, 20)
    print(f"CLASSIFIER COMPETITION RESULTS {trainee} v{version} prev{won_over_prev} VS {opponent_model}: ", winners)


    return losing_versions


def clone_new_version(prefix, from_version, to_version):
    m = tttp.TTTPlayer(model_name(prefix, "crosses", from_version))
    m.save_to_file(model_name(prefix, "crosses", to_version))

    m = tttp.TTTPlayer(model_name(prefix, "zeroes", from_version))
    m.save_to_file(model_name(prefix, "zeroes", to_version))

# --------------------------------------------
def main():
    prefix = args.save_to_model
    
    version = 0
    m_crosses = tttp.TTTPlayer()
    m_crosses.save_to_file(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer()
    m_zeroes.save_to_file(model_name(prefix, "zeroes", version))

    clone_new_version(prefix, 0, 1)
    version = 1
    epoch = 0
    trainee = "crosses"
    
    while True:

        # Compete and check if student wins now.
        losing_versions = versioned_competition(trainee, version, prefix)

        student_won = version not in losing_versions
        #student_won = len(losing_versions) == 0
        if student_won:
           print(f"VICTORY!!! STUDENT {trainee}.v{version} WON!")
           # If it does, update the version, and start training the other player
           trainee = "crosses" if trainee == "zeroes" else "zeroes"
           if trainee == "crosses":
               # Increment current version and copy last models, they will be trained next
               # Copy last model into a new version
               clone_new_version(prefix, version, version + 1)
               version += 1
        
           print(f"VICTORY!!! NOW STARTING TO TRAIN {trainee}.v{version}")
           print("\n\n")
           epoch = 0
           


        epoch += 1
        #versions_to_train_on = losing_versions
        versions_to_train_on = [version-1]
        train_single_epoch(epoch, prefix, version, trainee, versions_to_train_on)


if __name__ == "__main__":
    main()

