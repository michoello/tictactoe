from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys
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

def generate_playing_batch(num_games, m_crosses, m_zeroes):

    g = game.Game(m_crosses, m_zeroes)

    boards, values = [], []

    for i in range(num_games):
        steps, value = g.play_game(0.5, 2)
        for step in steps:
            boards.append(step.board.board)
            train_reward = [(step.reward + 1) / 2]
            values.append(train_reward)

    return boards, values


def generate_dumb_batch(num_boards, m_crosses, m_zeroes):
    #sum_weights = sum(value_weights)
    #boards_needed = [int(wei / sum_weights * num_boards) for wei in value_weights]
    boards_needed = [3 for _ in range(10)]
    print("BOARDS_NEEDED: ", boards_needed)
    outboards, outvalues = [], []
    num_games_played = 0

    while sum(boards_needed) > 0:
        num_games_played += 1
        boards, values = generate_playing_batch(1, m_crosses, m_zeroes)
        for board, value in zip(boards, values):
            value_bucket = int(value[0] * 0.999 * 10)
            if boards_needed[value_bucket] > 0:
                outboards.append(board)
                outvalues.append(value)
                boards_needed[value_bucket] -= 1
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


def train_single_epoch(epoch, m_crosses, m_zeroes, m_student, prefix, version, trainee):
    train_boards, train_values = generate_dumb_batch(32, m_crosses, m_zeroes)

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


def model_name(prefix, trainee, version):
   return f"{prefix}-{trainee}-{version}.json"


# Returns true if student wins over previous version
# TODO: check ALL prev versions victory
def versioned_competition(trainee, m_student, version, prefix):

    if trainee == "crosses":
        opponent = "zeroes"
        compete_version = version - 1
    else:
        opponent = "crosses"
        compete_version = version

    for v in range(0, compete_version + 1):
        opponent_model = model_name(prefix, opponent, v)
        m_opponent = tttp.TTTPlayer(opponent_model)

        if trainee == "crosses":
           winners = game.competition(m_student, m_opponent, 20)
        else:
           winners = game.competition(m_opponent, m_student, 20)
        print(f"VERSIONED COMPETITION RESULTS {trainee} VS {opponent_model}: ", winners)

        if v == compete_version:
           if trainee == "zeroes" and winners[-1] > 11:
               print(f"STUDENT {trainee}.v{version} WON over {opponent_model}!!!")
               return True
        
           if trainee == "crosses" and winners[1] > 11:
               print(f"STUDENT {trainee}.v{version} WON over {opponent_model}!!!")
               return True
           return False
    print("AAAAAAAAAAAA SHOULD NOT BE HERE")




# --------------------------------------------
def main():
    m_student: Any = tttp.TTTPlayer()
    if args.init_model is not None:
        print(f"Init player model: {args.init_model}")
        m_student.load_from_file(args.init_model)

    prefix = args.save_to_model
    
    version = 0
    m_crosses = tttp.TTTPlayer()
    m_crosses.save_to_file(model_name(prefix, "crosses", version))
    m_zeroes = tttp.TTTPlayer()
    m_crosses.save_to_file(model_name(prefix, "zeroes", version))

    version = 1
    epoch = 0
    trainee = "crosses"

    
    while True:
        # If it does, save the model version, and start training the other player
        model_file = model_name(prefix, trainee, version)

        epoch += 1
        print("-------------------------------------------------")
        print(f"TRAINING {model_file} EPOCH {epoch}")

        train_single_epoch(epoch, m_crosses, m_zeroes, m_student, prefix, version, trainee)
        m_student.save_to_file(model_file) 

        # Now we will generate next batch using our student as one of the players
        if trainee == "zeroes":
           m_zeroes = m_student
        else:
           m_crosses = m_student

        student_won = versioned_competition(trainee, m_student, version, prefix)

        # Compete and check if student wins now.
        if student_won:
           # If it does, update the version, and start training the other player
           old_trainee = trainee

           if trainee == "crosses":
               trainee = "zeroes"
               m_student = m_zeroes
           else:
               trainee = "crosses"
               m_student = m_crosses
               version += 1

           epoch = 0

           print(f"VICTORY!!! STUDENT {old_trainee} WON! NOW STARTING TO TRAIN {trainee} VERSION {version}")



if __name__ == "__main__":
    main()

