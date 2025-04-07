#
#  2025-03-16 !!! CURRENTLY UNDER CONSTRUCTION  !!!
# Run model training for temporal learning classifier
#
python train_by_playing.py --init_model models/model_playing_128.json --save_to_model models/model_playing_128_2.json



# Supposedly it should win here, but not yet - losing almost all the games :(
python generate_games.py play_many_games  models/model_victory_only.json models/model_playing.json



#
# More or less stable Commands to run
#


#
# Running unittests, including training session with results probabilistic evaluation
#
./testme.sh

```
pip install -e .
```

#
# Run model training classifier for victory board
# Dumps initial weight (random) in first file, final (best) in second file
#
python train_victory_classifier.py models/model_initial.json models/model_victory_only.json

#
# Load two models and play a single game between them
# First model crosses, second zeroes
#
python generate_games.py play_single_game  models/model_initial.json models/model_victory_only.json


#
# Play many games between two models and output stats
# First model crosses, second zeroes
#
python generate_games.py play_many_games  models/model_initial.json models/model_victory_only.json








# ------------------------


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
