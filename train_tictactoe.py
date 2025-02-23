from lib import game
from lib import ttt_classifier as ttt
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

initial_model_dump = sys.argv[1]
final_model_dump = sys.argv[2]

m = ttt.TTTClass()
m.save_to_file(initial_model_dump)


best_test_loss = 10 ** 1000
for epoch in range(100):

    test_boards, test_winners = game.generate_batch(TEST_BATCH_SIZE) 
    train_boards, train_winners = game.generate_batch(TRAINING_BATCH_SIZE) 

    for i in range(100):
      
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
    
      if test_loss < best_test_loss:
          m.save_to_file(final_model_dump)
          best_test_loss = test_loss
      else:
          break
