from lib import game
from lib import ttt_classifier as tttc
from lib import ttt_player as tttp
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

final_model_dump = sys.argv[1]


def generate_playing_batch(num_games, weighted=False):

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
      if weighted and (reward > 0.8 or reward < -0.8):
          for i in range(5):
             boards.append(board)
             values.append(train_reward)
  
  return boards, values


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

train_iterations = 5
best_test_loss = 10 ** 1000
for epoch in range(1000):

    # Gradient descent
    train_boards, train_values = generate_playing_batch(20, True)
    for i in range(train_iterations):
      
        # Backward pass
        train_loss = 0
        for board, value in zip(train_boards, train_values):
            m.x.set(board)
            m.y.set([value]) 

            m.loss.dif()
            m.apply_gradient()
    
            loss = m.loss.val()
            train_loss = train_loss + loss[0][0]

        train_loss = calc_loss(m, train_boards, train_values)
        test_loss = calc_loss(m, test_boards, test_values)
        print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")

    train_loss_buckets = calc_loss_buckets(m, train_boards, train_values)
    print(f"\nTRAIN LOSS BUCKETS: {train_loss_buckets}")
    test_loss_buckets = calc_loss_buckets(m, test_boards, test_values)
    print(f"\nTEST LOSS BUCKETS: {test_loss_buckets}\n\n")


   
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

    #sys.exit(0)


    if test_loss < best_test_loss:
      print(f"EPOCH {epoch}: SAVING!")
      m.save_to_file(final_model_dump)
      best_test_loss = test_loss
