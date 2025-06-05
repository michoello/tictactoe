from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys
import os
from typing import Any


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
        steps, winner = g.play_game(0.5, 2)
        if trainee == "zeroes" and winner == -1:
            i += 1
            continue
        if trainee == "crosses" and winner == 1:
            i += 1
            continue
        for step in steps:
            boards.append(step.board.board)
            train_reward = [(step.reward + 1) / 2]
            values.append(train_reward)
        break
    print(f"WHILE GENERATING BATCH FOR TRAINING, THE NUMBER OF GAMES {trainee} WON IS {i}")
    return boards, values


def generate_dumb_batch(num_boards, m_crosses, m_zeroes, trainee):
    outboards, outvalues = [], []
    num_games_played = 0

    i = 0
    while i < num_boards: 
        num_games_played += 1
        boards, values = generate_playing_batch(1, m_crosses, m_zeroes, trainee)
        for board, value in zip(boards, values):
            value_bucket = int(value[0] * 0.999 * 10)
            outboards.append(board)
            outvalues.append(value)
            i += 1

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


def train_single_epoch(epoch, prefix, version, trainee):
    print("-------------------------------------------------")
    print(f"TRAINING {trainee} {version} EPOCH {epoch}")

    m_crosses = tttp.TTTPlayer(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer(model_name(prefix, "zeroes", version))

    m_student = m_crosses if trainee == "crosses" else m_zeroes

    train_boards, train_values = generate_dumb_batch(32, m_crosses, m_zeroes, trainee)

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
    # Play against classifier
    #
    opponent_model = "models/model_victory_only.json"
    m_opponent = tttc.TTTClass(opponent_model)
    if trainee == "crosses":
        winners = game.competition(m_student, m_opponent, 20)
    else:
        winners = game.competition(m_opponent, m_student, 20)
    print(f"CLASSIFIER COMPETITION RESULTS {trainee} VS {opponent_model}: ", winners)

    #
    # Play against previous versions
    #
    for v in range(0, version + 1):
        opponent_model = model_name(prefix, opponent, v)
        m_opponent = tttp.TTTPlayer(opponent_model)

        if trainee == "crosses":
           winners = game.competition(m_student, m_opponent, 20)
        else:
           winners = game.competition(m_opponent, m_student, 20)
        print(f"VERSIONED COMPETITION RESULTS {trainee} VS {opponent_model}: ", winners)

        if v == version:
           if trainee == "zeroes" and winners[-1] > 11:
               print(f"STUDENT {trainee}.v{version} WON over {opponent_model}!!!")
               return True
        
           if trainee == "crosses" and winners[1] > 11:
               print(f"STUDENT {trainee}.v{version} WON over {opponent_model}!!!")
               return True
           return False
    print("AAAAAAAAAAAA SHOULD NOT BE HERE")
    return False


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

        epoch += 1
        train_single_epoch(epoch, prefix, version, trainee)

        # Compete and check if student wins now.
        student_won = versioned_competition(trainee, version, prefix)
        if student_won:
           # If it does, update the version, and start training the other player
           old_trainee = trainee

           if trainee == "crosses":
               trainee = "zeroes"
           else:
               trainee = "crosses"

               # Increment current version and copy last models, they will be trained next
               # Copy last model into a new version
               clone_new_version(prefix, version, version + 1)
               version += 1
               #m = tttp.TTTPlayer(model_name(prefix, "crosses", version - 1))
               #m.save_to_file(model_name(prefix, "crosses", version))
               #m = tttp.TTTPlayer(model_name(prefix, "zeroes", version - 1))
               #m.save_to_file(model_name(prefix, "zeroes", version))
        
           epoch = 0

           print(f"VICTORY!!! STUDENT {old_trainee} WON! NOW STARTING TO TRAIN {trainee} VERSION {version}")
           print("\n\n")



if __name__ == "__main__":
    main()

