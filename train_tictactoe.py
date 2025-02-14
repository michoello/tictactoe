from lib import game
from lib import ttt_classifier as ttt
import sys

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

def gradient_backpropagation(m, x, y, epoch):

    # Fit in the inputs and labels
    m.x.set(x)
    m.y.set(y) 

    # Backward pass
    m.loss.dif()

    # Update weights and biases
    #alpha = 0.1 if epoch < 5 else 0.001
    #alpha = 0.1 
    alpha = 0.01 if epoch < 10 else 0.001

    m.w1.appl(alpha) 
    m.b1.appl(alpha) 
    m.w2.appl(alpha)
    m.b2.appl(alpha)
    #m.w3.appl(alpha)
    #m.b3.appl(alpha)


m = ttt.TTTClass()


model_dump = m.loss.save()
with open("models/model_initial.json", "w") as file:
    file.write(model_dump)


#with open("models/model_initial.json", "r") as file:
#    model_dump = file.read() 
#m.loss.load(model_dump)


test_boards, test_winners = game.generate_batch(TEST_BATCH_SIZE) 


best_test_loss = 10 ** 1000
for epoch in range(100):

    train_boards, train_winners = game.generate_batch(TRAINING_BATCH_SIZE) 

    for i in range(100):
      
      train_loss = 0
      for board, winner in zip(train_boards, train_winners):
        gradient_backpropagation(m, board, [winner], epoch)
    
        loss, prediction = m.loss.val(), m.prediction.val()
        #print("TRAIN PREDICTION:", prediction)
        train_loss = train_loss + loss[0][0]
    
      #sys.exit(1)

      test_loss = 0
      for board, winner in zip(test_boards, test_winners):
          m.x.set(board)
          m.y.set([winner]) 
    
          loss, prediction = m.loss.val(), m.prediction.val()
          #print("TEST PREDICTION:", prediction)
          test_loss = test_loss + loss[0][0]

      train_loss = train_loss/len(train_boards)
      test_loss = test_loss/len(test_boards)
      print(f"EPOCH {epoch}/{i}: Train loss={train_loss}\t\tTest loss = {test_loss}")
    
      if test_loss < best_test_loss:
          model_dump = m.loss.save()
          with open("models/model_trained.json", "w") as file:
              file.write(model_dump)
          best_test_loss = test_loss
    
      else:
          break
    
