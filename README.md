Here it comes.
And one more line.

# Commands to run
```
pip install -e .

#
# Run model training
# Dumps initial weight (random) in first file, final (best) in second file
#
python train_tictactoe.py models/model_initial.json models/model_trained.json

#
# Load two models and play a single game between them
# First model crosses, second zeroes
#
python generate_games.py play_single_game  models/model_initial.json models/model_trained.json


#
# Play many games between two models and output stats
# First model crosses, second zeroes
#
python generate_games.py play_single_game  models/model_initial.json models/model_trained.json



#
# Other modes
#
# To generate many random boards with stats
python generate_games.py random_games

# To generate random single game
python generate_games.py gen_random_game


# To generate many games
python generate_games.py many_games



# To run unittests
python -m unittest discover -s tests
# or
./testme.sh
```
