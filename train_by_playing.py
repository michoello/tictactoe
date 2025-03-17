from lib import game
from lib import ttt_classifier as ttt
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

final_model_dump = sys.argv[1]

#m.save_to_file(initial_model_dump)


def generate_playing_batch(num_games):

  m_crosses = ttt.TTTClass("models/model_victory_only.json")
  m_zeroes = ttt.TTTClass("models/model_victory_only.json")
  g = game.Game(m_crosses, m_zeroes)

  boards, values = [], []

  for i in range(num_games):
    steps, winner = g.play_game(0.1)
    for step_no, (_, board, ply, x, y, reward) in enumerate(steps):
      boards.append(board)
      values.append([(reward+1)/2])
  
  return boards, values

m = ttt.TTTClass()

best_test_loss = 10 ** 1000
for epoch in range(100):

    train_boards, train_winners = generate_playing_batch(25)
    test_boards, test_winners = generate_playing_batch(1)

    for i in range(200):
      
      train_loss = 0
      for board, winner in zip(train_boards, train_winners):
        m.x.set(board)
        m.y.set([winner]) 

        # Backward pass
        m.loss.dif()
        m.apply_gradient()
    
        loss = m.loss.val()
        train_loss = train_loss + loss[0][0]

      test_loss = 0
      for board, winner in zip(test_boards, test_winners):
          m.x.set(board)
          m.y.set([winner]) 
    
          loss = m.loss.val()
          test_loss = test_loss + loss[0][0]

      train_loss = train_loss/len(train_boards)
      test_loss = test_loss/len(test_boards)
      print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")
    
    if epoch == 0:
      #for board, winner in zip(test_boards, test_winners):
      for board, winner in zip(train_boards, train_winners):
        m.x.set(board)
        m.y.set([winner]) 
        loss = m.loss.val()
        prediction = m.prediction.val()

        game.print_board(board)
        print(f"WINNER: {winner}, PREDICTION {prediction} LOSS {loss}")

      break


    if test_loss < best_test_loss:
      print(f"EPOCH {epoch}: SAVING!")
      m.save_to_file(final_model_dump)
      best_test_loss = test_loss
