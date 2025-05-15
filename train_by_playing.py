from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys
from typing import Any


import argparse

parser = argparse.ArgumentParser(description="Train your model")

parser.add_argument("--init_model", type=str, help="Path to the initial model file")
parser.add_argument("--save_to_model", type=str, help="Path to save the trained model")
#parser.add_argument("--trainee", choices=["crosses", "zeroes"], help="Choose whom to train: 'crosses' or 'zeroes'")


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


def generate_balanced_batch(num_boards, value_weights, m_crosses, m_zeroes):
    sum_weights = sum(value_weights)
    boards_needed = [int(wei / sum_weights * num_boards) for wei in value_weights]
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


def calc_loss_buckets(m, boards, values):
    loss_buckets = [[0, 0] for _ in range(10)]
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value])
        loss = m.loss.val()

        loss_bucket = int(value[0] * 0.999 * 10)
        loss_buckets[loss_bucket][0] += loss[0][0]
        loss_buckets[loss_bucket][1] += 1

    for bucket in range(10):
        tt = loss_buckets[bucket]
        loss_buckets[bucket] = tt[0] / tt[1] if tt[1] != 0 else 0.1
    return loss_buckets


def calc_loss(m, boards, values):
    sum_loss = 0
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value])

        loss = m.loss.val()
        sum_loss = sum_loss + loss[0][0]

    return sum_loss / len(boards)


def train_single_epoch(epoch, m_crosses, m_zeroes, m_student):
    test_boards, test_values = generate_playing_batch(100, m_crosses, m_zeroes)

    test_loss_buckets = calc_loss_buckets(m_student, test_boards, test_values)
    print(f"\nTEST LOSS BUCKETS: ", [round(l, 2) for l in test_loss_buckets])

    train_boards, train_values = generate_balanced_batch(
        32, test_loss_buckets, m_crosses, m_zeroes
    )
    train_boards_b, train_values_b = train_boards, train_values

    train_loss_buckets = calc_loss_buckets(m_student, train_boards, train_values)
    print(
        f"\nTRAIN LOSS BUCKETS: ",
        [round(l, 2) if l is not None else "None" for l in train_loss_buckets],
    )

    # Backward pass
    train_iterations = 25
    for i in range(train_iterations):
        train_loss = 0
        for board, value in zip(train_boards_b, train_values_b):
            m_student.x.set(board)
            m_student.y.set([value])

            m_student.loss.dif()
            m_student.apply_gradient()

            loss = m_student.loss.val()
            train_loss = train_loss + loss[0][0]

        train_loss = calc_loss(m_student, train_boards, train_values)
        test_loss = calc_loss(m_student, test_boards, test_values)
        print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")

# Returns true if student wins
def competition(m_crosses, m_zeroes, trainee):
    winners = game.competition(m_crosses, m_zeroes, 20)
    print("COMPETITION RESULTS: ", winners)

    if trainee == "zeroes" and winners[-1] > winners[1] + 2:
       return True
        
    if trainee == "crosses" and winners[1] > winners[-1] + 2:
       return True
    return False



def model_name(prefix, trainee, version):
   return f"{prefix}-{trainee}-{version}.json"


def versioned_competition(trainee, m_student, version, prefix):
    opponent = "crosses" if trainee == "zeroes" else "zeroes"
    for v in range(1, version):
        m_opponent = tttp.TTTPlayer(model_name(prefix, opponent, v))
        if trainee == "zeroes":
           m_crosses, m_zeroes = m_opponent, m_student
        else:
           m_crosses, m_zeroes = m_student, m_opponent

        winners = game.competition(m_opponent, m_zeroes, 20)
        print(f"COMPETITION RESULTS: for version {v} ", winners)



# --------------------------------------------
def main():
    m_student: Any = tttp.TTTPlayer()
    if args.init_model is not None:
        print(f"Init player model: {args.init_model}")
        m_student.load_from_file(args.init_model)
    
    m_crosses = tttc.TTTClass("models/model_victory_only.json")
    m_zeroes = tttc.TTTClass("models/model_victory_only.json")

    trainee = "zeroes"
    version = 1
    epoch = 0

    prefix = args.save_to_model
    
    while True:
        epoch += 1
        print("-------------------------------------------------")
        print(f"TRAINING {trainee} VERSION {version} EPOCH {epoch}")

        train_single_epoch(epoch, m_crosses, m_zeroes, m_student)

        # Now we will generate next batch using our student as one of the players
        if trainee == "zeroes":
           m_zeroes = m_student
        if trainee == "crosses":
           m_crosses = m_student


        versioned_competition(trainee, m_student, version, prefix)

        # Compete and check if student wins now.
        student_won = competition(m_crosses, m_zeroes, trainee)
        if student_won:
           # If it does, save the model version, and start training the other player
           model_file = model_name(prefix, trainee, version)

           print(f"STUDENT {trainee} WON! SAVING {model_file} AND SWITCHING")
           m_student.save_to_file(model_file) 

           trainee = "zeroes" if trainee == "crosses" else "crosses"
           if trainee == "zeroes":
              version += 1
           epoch = 0

           if trainee == "crosses":
               # TODO: make it straight, too ugly now
               if version == 1:
                   m_student = tttp.TTTPlayer()
               else:
                   m_student = m_crosses
           else:
               m_student = m_zeroes
           #m_student = m_crosses if trainee == "crosses" else m_zeroes



if __name__ == "__main__":
    main()

