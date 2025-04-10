import unittest
from lib import ml
from lib import game 
import tempfile
from lib import ttt_classifier as ttt


def roughlyEqual(m1, m2):
    if all([round(a, 2) == round(b, 2) for row_a, row_b in zip(m1, m2) for a, b in zip(row_a, row_b)]):
        return True
    print(f"Arrays not equal:\n{m1}\n{m2}")
    return False


class TestTrainingCycle(unittest.TestCase):

    def test_omg(self):
        x = [[1, 2]] 
        w1 = [[3, 4, 5], [6, 7, 8]] 
        w_wrong = [[3, 4, 5]] 
        
        xx = ml.BB(x)
        ww1 = ml.BB(w1)

        yy = xx @ ww1

        assert xx.dims() == [1, 2]
        assert ww1.dims() == [2, 3], f"Actual value is {ww1.dims()}"
        assert yy.dims() == [1, 3], f"Actual value is {yy.dims()}"


        assert yy.val() == [ [ 15, 18, 21 ] ]

        ww_wrong = ml.BB(w_wrong)
        with self.assertRaises(ValueError):
           y_wrong = xx @ ww_wrong

        b = [[1, 2, 3]]

        yy1 = yy + ml.BB(b)

        assert yy1.val() == [[16, 20, 24]], f"actual value is {yy1.val()}"


        # update x values
        xx.set([[11,22]])
        self.assertEqual(xx.val(), [[11,22]])
        # Check that yy (which is xx @ ww) is also updated
        self.assertEqual(yy.val(), [[165, 198, 231]])


    def test_training_classifier_and_game(self):
        init_model = tempfile.mktemp()
        trained_model = tempfile.mktemp()

        m = ttt.TTTClass()
        m.save_to_file(init_model)
        
        print("Training")
        best_test_loss = 10 ** 1000
        test_boards, test_winners = game.generate_batch(20) 
        for epoch in range(50):
            train_boards, train_winners = game.generate_batch(20) 
        
            for i in range(10):
              
              for board, winner in zip(train_boards, train_winners):
                m.x.set(board)
                m.y.set([winner]) 
                m.loss.dif()
                m.apply_gradient()
        
              test_loss = 0
              for board, winner in zip(test_boards, test_winners):
                m.x.set(board)
                m.y.set([winner]) 
                test_loss = test_loss + m.loss.val()[0][0]
            
              if test_loss < best_test_loss:
                  m.save_to_file(trained_model)
                  best_test_loss = test_loss
        
            if epoch % 5 == 0:
                print(f"{epoch/50*100}% - test_loss {test_loss}")


        print("Playing")
        random_model = ttt.TTTClass(init_model)
        trained_model = ttt.TTTClass(trained_model)
        winners = {
          'random': 0,
          'trained': 0,
          'tie': 0
        }

        for f in range(50):
           if f % 2 == 0:
              # Trained model plays zeroes
              g = game.Game(random_model, trained_model)
              trained_player = -1 
           else:
              g = game.Game(trained_model, random_model)
              trained_player = 1

           _, winner = g.play_game(0.3)
           if winner == 0:
               winners['tie'] += 1
           elif winner == trained_player:
               winners['trained'] += 1
           else:
               winners['random'] += 1

        print("WINNERS", winners)
        # This sometimes fails. TODO: find a way to pass reliably
        self.assertGreater(winners['trained'], winners['random'])





if __name__ == "__main__":
    unittest.main()
