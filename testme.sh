./run_mypy.sh || { echo "MyPy failed"; exit 1; }

ARG=$1

if [ "$ARG" == "mcts" ]; then
    echo "Running mcts tests"
    python3 -m unittest tests.test_mcts --verbose
elif [ "$ARG" == "game" ]; then
    echo "Running game tests"
    python3 -m unittest tests.test_game --verbose
else
    echo "Running all unittests"
    python3 -m unittest discover -s tests --verbose
fi

# To test only game training:
# python3 -m unittest --verbose tests.test_game.TestTrainingCycle.test_training_classifier_and_game
#
# Only replay buffer
# python3 -m unittest --verbose tests.test_replay_buffer.TestReplayBuffer
