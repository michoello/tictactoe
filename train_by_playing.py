from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys
from typing import Any


import argparse

parser = argparse.ArgumentParser(description="Train your model")

parser.add_argument('--init_model', type=str, help='Path to the initial model file')
parser.add_argument('--save_to_model', type=str, help='Path to save the trained model')

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
      train_reward = [(step.reward+1)/2]
      values.append(train_reward)
  
  return boards, values


def generate_balanced_batch(num_boards, value_weights, m_crosses, m_zeroes):
    sum_weights = sum(value_weights)
    boards_needed  = [int(wei/sum_weights*num_boards) for wei in value_weights]
    print("BOARDS_NEEDED: ", boards_needed)
    outboards, outvalues = [], []
    num_games_played = 0

    while sum(boards_needed) > 0:
       num_games_played += 1
       boards, values = generate_playing_batch(1, m_crosses, m_zeroes)
       for board, value in zip(boards, values):
           value_bucket = int(value[0]*0.999 * 10)
           if boards_needed[value_bucket] > 0:
                outboards.append(board)
                outvalues.append(value)
                boards_needed[value_bucket] -= 1
    print("BALANCED BATCH READY: num_boards=", len(outboards), "games_played=", num_games_played)
    return outboards, outvalues          


def calc_loss_buckets(m, boards, values):
    loss_buckets = [ [0, 0] for _ in range(10)]
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value]) 
        loss = m.loss.val()
          
        loss_bucket = int(value[0]*0.999 * 10)
        loss_buckets[loss_bucket][0] += loss[0][0]
        loss_buckets[loss_bucket][1] += 1

    for bucket in range(10):
        tt = loss_buckets[bucket] 
        loss_buckets[bucket] = tt[0] / tt[1] if tt[1] != 0 else None
    return loss_buckets


def calc_loss(m, boards, values):
    sum_loss = 0
    for board, value in zip(boards, values):
        m.x.set(board)
        m.y.set([value]) 
   
        loss = m.loss.val()
        sum_loss = sum_loss + loss[0][0]

    return sum_loss / len(boards)


# --------------------------------------------

m: Any = tttp.TTTPlayer()
if args.init_model is not None:
    print(f"Init player model: {args.init_model}")
    m.load_from_file(args.init_model)


train_iterations = 25
best_test_loss = 10 ** 1000

m_crosses = tttc.TTTClass("models/model_victory_only.json")
m_zeroes = tttc.TTTClass("models/model_victory_only.json")

for epoch in range(1000):

    test_boards, test_values = generate_playing_batch(100, m_crosses, m_zeroes)

    test_loss_buckets = calc_loss_buckets(m, test_boards, test_values)
    print(f"\nTEST LOSS BUCKETS: ", [round(l, 2) for l in test_loss_buckets])

    train_boards, train_values = generate_balanced_batch(32, test_loss_buckets, m_crosses, m_zeroes)
    train_boards_b, train_values_b = train_boards, train_values


    train_loss_buckets = calc_loss_buckets(m, train_boards, train_values)
    print(f"\nTRAIN LOSS BUCKETS: ", [round(l, 2) if l is not None else "None" for l in train_loss_buckets])

    for i in range(train_iterations):
      
        # Backward pass
        train_loss = 0
        for board, value in zip(train_boards_b, train_values_b):
            m.x.set(board)
            m.y.set([value]) 

            m.loss.dif()
            m.apply_gradient()
    
            loss = m.loss.val()
            train_loss = train_loss + loss[0][0]

        # Batch gradient application. Does not help much
        # m.apply_gradient()

        train_loss = calc_loss(m, train_boards, train_values)
        test_loss = calc_loss(m, test_boards, test_values)
        print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")
   

    # Print extended stats each 10 epochs
    if epoch % 10 == 0:
      #for board, value in zip(test_boards, test_values):
      for board, value in zip(train_boards, train_values):
        m.x.set(board)
        m.y.set([value]) 
        loss = m.loss.val()
        prediction = m.prediction.val()

        #game.Board(board).print_board()
        #print(f"WINNER: {value}, PREDICTION {prediction} LOSS {loss}")

    if test_loss < best_test_loss and args.save_to_model is not None:
      print(f"EPOCH {epoch}: SAVING loss {test_loss} to {args.save_to_model}")
      m.save_to_file(args.save_to_model)
      best_test_loss = test_loss


    winners = game.competition(m_crosses, m, 20)
    print("COMPETITION RESULTS: ", winners)
    if winners[-1] > winners[1]:
       m.save_to_file(args.save_to_model)
       sys.exit(0)    
    



    # Now we will generate next batch using our student as one of the players
    m_zeroes = m
