import unittest
from lib import ml
from lib import game
import tempfile
from lib import ttt_classifier as ttt
from utils import roughlyEqual
from utils import SimpleRNG
from unittest.mock import patch


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

        assert yy.val() == [[15, 18, 21]]

        ww_wrong = ml.BB(w_wrong)
        with self.assertRaises(ValueError):
            y_wrong = xx @ ww_wrong

        b = [[1, 2, 3]]

        yy1 = yy + ml.BB(b)

        assert yy1.val() == [[16, 20, 24]], f"actual value is {yy1.val()}"

        # update x values
        xx.set([[11, 22]])
        self.assertEqual(xx.val(), [[11, 22]])
        # Check that yy (which is xx @ ww) is also updated
        self.assertEqual(yy.val(), [[165, 198, 231]])

    def test_training_classifier_and_game(self):
        rng = SimpleRNG(seed=45)
        with patch("random.random", new=rng.random), patch(
            "random.randint", new=rng.randint
        ), patch("random.choice", new=rng.choice), patch(
            "random.shuffle", new=rng.shuffle
        ):
            init_model = tempfile.mktemp()
            trained_model = tempfile.mktemp()

            m = ttt.TTTClass()
            m.save_to_file(init_model)

            print("Training")
            best_test_loss = 10**1000
            test_boards, test_winners = game.generate_batch(20)

            total_epochs = 100
            good_enough_reached = False
            for epoch in range(total_epochs):
                train_boards, train_winners = game.generate_batch(20)

                for i in range(20):

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

                if epoch % 5 == 0 or best_test_loss < 2.9:
                    print(f"{epoch/total_epochs*100}% - test_loss {test_loss}")
                    if best_test_loss < 2.9:
                        # good enough - sometimes it does not re
                        good_enough_reached = True
                        break

            if not good_enough_reached:
                print("!!!!!!!!!!!!!")
                print(
                    "Did not train till good test loss. TODO: find out why this is happening (bad init weights?) "
                )
                print("Test will prob fail. Rerun it then")

            print("Playing...")
            random_model = ttt.TTTClass(init_model)
            trained_model = ttt.TTTClass(trained_model)

            # ctw = crosses_trained_winners
            ctw = game.competition(trained_model, random_model, 20)
            print("Trained crosses WINNERS cross:", ctw[1], " zero:", ctw[-1])
            self.assertGreater(ctw[1], ctw[-1])

            # ztw = zeroes_trained_winners
            ztw = game.competition(random_model, trained_model, 20)
            print("Trained zeroes WINNERS cross:", ztw[1], " zero:", ztw[-1])
            self.assertLess(ztw[1], ztw[-1])


if __name__ == "__main__":
    unittest.main()
