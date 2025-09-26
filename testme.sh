python3 -m unittest discover -s tests --verbose

# To test only game training:
# python3 -m unittest --verbose tests.test_game.TestTrainingCycle.test_training_classifier_and_game
#
# Only replay buffer
# python3 -m unittest --verbose tests.test_replay_buffer.TestReplayBuffer
