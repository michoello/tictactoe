from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

final_model_dump = sys.argv[1]


def generate_playing_batch(num_games):

  m_crosses = tttc.TTTClass("models/model_victory_only.json")
  m_zeroes = tttc.TTTClass("models/model_victory_only.json")
  g = game.Game(m_crosses, m_zeroes)

  boards, values = [], []

  for i in range(num_games):
    steps, value = g.play_game(0.3)
    for step_no, (_, board, ply, x, y, reward) in enumerate(steps):
      boards.append(board)
      train_reward = [(reward+1)/2]
      values.append(train_reward)
  
  return boards, values


def generate_balanced_batch(num_boards, value_weights):
    sum_weights = sum(value_weights)
    boards_needed  = [int(wei/sum_weights*num_boards) for wei in value_weights]
    print("BOARDS_NEEDED: ", boards_needed)
    outboards, outvalues = [], []
    num_games_played = 0
    while sum(boards_needed) > 0:
       num_games_played += 1
       boards, values = generate_playing_batch(1)
       for board, value in zip(boards, values):
           value_bucket = int(value[0]*0.999 * 10)
           if boards_needed[value_bucket] > 0:
                outboards.append(board)
                outvalues.append(value)
                boards_needed[value_bucket] -= 1
    print("BALANCED BATCH READY: num_boards=", len(outboards), "games_played=", num_games_played)
    return outboards, outvalues          



def balance_batch(train_board, train_values):
    train_boards_b, train_values_b = [], []
    for board, value in zip(train_boards, train_values):
       count = 5 if (value[0] > 0.9 or value[0] < 0.1) else 1
       for i in range(count):
           train_boards_b.append(board)
           train_values_b.append(value)

    return train_boards_b, train_values_b


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

m = tttp.TTTPlayer()

test_boards, test_values = generate_playing_batch(10)

train_iterations = 25
best_test_loss = 10 ** 1000
for epoch in range(1000):

    test_loss_buckets = calc_loss_buckets(m, test_boards, test_values)
    print(f"\nTEST LOSS BUCKETS: ", [round(l, 2) for l in test_loss_buckets])

    # !!!
    train_boards, train_values = generate_balanced_batch(100, test_loss_buckets)
    train_boards_b, train_values_b = train_boards, train_values


    # Gradient descent
    #train_boards, train_values = generate_playing_batch(20)
    #train_boards_b, train_values_b = balance_batch(train_boards, train_values)

    #sys.exit(0)

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

        game.print_board(board)
        print(f"WINNER: {value}, PREDICTION {prediction} LOSS {loss}")



    if test_loss < best_test_loss:
      print(f"EPOCH {epoch}: SAVING!")
      m.save_to_file(final_model_dump)
      best_test_loss = test_loss
