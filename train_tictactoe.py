from lib import game
from lib import ttt_classifier as ttt
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

def gradient_backpropagation(x, y, epoch):

    # Fit in the inputs and labels
    ttt.x.set(x)
    ttt.y.set(y) 

    # Backward pass
    ttt.loss.dif()

    # Update weights and biases
    #alpha = 0.1 if epoch < 5 else 0.001
    alpha = 0.1 
    #alpha = 1

    ttt.w1.appl(alpha) 
    ttt.b1.appl(alpha) 
    ttt.w2.appl(alpha)
    ttt.b2.appl(alpha)
    ttt.w3.appl(alpha)
    ttt.b3.appl(alpha)


#model_dump = ttt.loss.save()
#with open("models/model_initial.json", "w") as file:
#    file.write(model_dump)


#with open("models/model_initial.json", "r") as file:
#    model_dump = file.read() 
#ttt.loss.load(model_dump)


test_boards, test_winners = game.generate_batch(TEST_BATCH_SIZE) 


best_test_loss = 10 ** 1000
for epoch in range(100):

    train_boards, train_winners = game.generate_batch(TRAINING_BATCH_SIZE) 

    for i in range(100000):
      
      train_loss = 0
      for board, winner in zip(train_boards, train_winners):
        gradient_backpropagation(board, [winner], epoch)
    
        loss, prediction = ttt.loss.val(), ttt.prediction.val()
        train_loss = train_loss + loss[0][0]
    
      test_loss = 0
      for board, winner in zip(test_boards, test_winners):
          ttt.x.set(board)
          ttt.y.set([winner]) 
    
          loss, prediction = ttt.loss.val(), ttt.prediction.val()
          test_loss = test_loss + loss[0][0]

      train_loss = train_loss/len(train_boards)
      test_loss = test_loss/len(test_boards)
      print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")
    
      if test_loss < best_test_loss:
          model_dump = ttt.loss.save()
          with open("models/model_trained.json", "w") as file:
              file.write(model_dump)
          best_test_loss = test_loss
    
      else:
          break
    
