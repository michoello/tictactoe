# 2025-04-16
# Updating model params does not help, it looks like something is missing
# Observation:
#  - The model does not recognize a state "one step from losing", when e.g. 3 crosses in a row, and we play zero.
#    The model MUST put zero in the cell where 4th cross would be set, even if it does not lead to winning.
#
# Ideas:
#  - Maybe we can choose best step not only "aggressive", but also "defensive", when another player is to close to victory?
#    The problem is that this logic "leaks" game rules into the playing, which is not good for generalization
#  - Instead, we can explore states before "loss", and generate more boards, but that leaks rules into training. Not good again
#  - We can introduce a feedback loop into training, by making the trained model play games, to make it explore its mistakes.
#


# 2025-04-14
# Updated playing script to support two different models. The player model still fails.
# TODO run single game and inspect weights
python generate_games.py --mode play_single_game \
   --zeroes_model classifier:models/model_victory_only.json \
   --crosses_model player:models/model_playing_mse_loss.json

python generate_games.py --mode play_many_games \
   --zeroes_model classifier:models/model_victory_only.json \
   --crosses_model player:models/model_playing_mse_loss.json


#
#  2025-04-11 !!! CURRENTLY UNDER CONSTRUCTION  !!!
# Ran model training for temporal learning classifier with MSE loss:
# Looking at the model output it looks better
#
python train_by_playing.py --save_to_model models/model_playing_mse_loss.json

#
#  2025-03-16
# Ran model training for temporal learning classifier:
# - adding layers or weights did not help much
# - batch gradient did not help at all
# - trying MSE now
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
python generate_games.py --mode play_single_game \
   --crosses_model classifier:models/model_initial.json \
   --zeroes_model classifier:models/model_victory_only.json 


#
# Play many games between two models and output stats
# First model crosses, second zeroes
#
python generate_games.py --mode play_many_games \
   --crosses_model classifier:models/model_initial.json \
   --zeroes_model classifier:models/model_victory_only.json 



# ------------------------


#
# Other modes
#
# To generate many random boards with stats
python generate_games.py --mode random_games

# To generate random single game
python generate_games.py --mode gen_random_game


# To generate many games
python generate_games.py --mode many_games



# To run unittests
python -m unittest discover -s tests
# or
./testme.sh
```
