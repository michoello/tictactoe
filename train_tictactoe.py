from lib import game
from lib import ttt_classifier as ttt
import sys
import math

TRAINING_BATCH_SIZE = 60
TEST_BATCH_SIZE = 30

def norm(matrix):
    return math.sqrt(sum(sum(x**2 for x in row) for row in matrix))


def gradient_backpropagation(m, x, y, epoch):

    # Fit in the inputs and labels
    m.x.set(x)
    m.y.set(y) 

    # Backward pass
    m.loss.dif()

    # Update weights and biases
    #alpha = 0.01 if epoch < 10 else 0.001


    # TODO: interesting! explore more
    w1norm = norm(m.w1.val())
    w1dnorm = norm(m.w1.dval())
    alpha = 0.01
    if w1dnorm / w1norm < 0.01:
        alpha = 0.1 
    if w1dnorm / w1norm > 1000:
        alpha = 0.001
    #print(norm(m.w2.val()))
    #print(norm(m.w2.dval()))
    #print(norm(m.b2.val()))
    #print(norm(m.b2.dval()))
    #print(norm(m.b1.val()))
    #print(norm(m.b1.dval()))
    #sys.exit(0)

    m.w1.appl(alpha) 
    m.b1.appl(alpha) 
    m.w2.appl(alpha)
    m.b2.appl(alpha)
    #m.w3.appl(alpha)
    #m.b3.appl(alpha)


m = ttt.TTTClass()
m.save_to_file("models/model_initial.json")

test_boards, test_winners = game.generate_batch(TEST_BATCH_SIZE) 


best_test_loss = 10 ** 1000
for epoch in range(100):

    train_boards, train_winners = game.generate_batch(TRAINING_BATCH_SIZE) 

    for i in range(100):
      
      train_loss = 0
      for board, winner in zip(train_boards, train_winners):
        gradient_backpropagation(m, board, [winner], epoch)
    
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
          m.save_to_file("models/model_trained.json")
          best_test_loss = test_loss
      else:
          break
