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
    steps, value = g.play_game(0.1)
    for step_no, (_, board, ply, x, y, reward) in enumerate(steps):
      boards.append(board)
      values.append([(reward+1)/2])
  
  return boards, values

m = tttp.TTTPlayer()

best_test_loss = 10 ** 1000
for epoch in range(100):

    train_boards, train_values = generate_playing_batch(25)
    test_boards, test_values = generate_playing_batch(1)

    for i in range(50):
      
      train_loss = 0
      for board, value in zip(train_boards, train_values):
        m.x.set(board)
        m.y.set([value]) 

        # Backward pass
        m.loss.dif()
        m.apply_gradient()
    
        loss = m.loss.val()
        train_loss = train_loss + loss[0][0]

      test_loss = 0
      for board, value in zip(test_boards, test_values):
          m.x.set(board)
          m.y.set([value]) 
    
          loss = m.loss.val()
          test_loss = test_loss + loss[0][0]

      train_loss = train_loss/len(train_boards)
      test_loss = test_loss/len(test_boards)
      print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")
    
    if epoch % 10 == 0:
      #for board, value in zip(test_boards, test_values):
      for board, value in zip(train_boards, train_values):
        m.x.set(board)
        m.y.set([value]) 
        loss = m.loss.val()
        prediction = m.prediction.val()

        game.print_board(board)
        print(f"WINNER: {value}, PREDICTION {prediction} LOSS {loss}")

      #break


    if test_loss < best_test_loss:
      print(f"EPOCH {epoch}: SAVING!")
      m.save_to_file(final_model_dump)
      best_test_loss = test_loss
