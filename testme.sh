#./run_mypy.sh || { echo "MyPy failed"; exit 1; }

ARG=$1

if [ -z "$ARG" ]; then
    echo "Running all unittests"
    python3 -m unittest discover -s tests --verbose
else
    echo "Running $ARG tests"
    python3 -m unittest tests.test_$ARG --verbose
fi

# To test only game training:
# python3 -m unittest --verbose tests.test_game.TestTrainingCycle.test_training_classifier_and_game
#
# Only replay buffer
# python3 -m unittest --verbose tests.test_replay_buffer.TestReplayBuffer
