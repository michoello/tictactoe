# 2025-05-15
Next I will try to train the model from zero level, without "classifier"
to compete against. If there will be progress, the "classifier"
model can be used as a test set -- i.e. something we know somewhat capable
of winning other zero-random model

# 2025-05-05
Multi iterational training. 
We start with zeroes as student.
Once a student wins over the teacher, we swap the roles. Now the teacher becomes the student, and previous student is used as a teacher.
So now we train crosses. And so on.
Kind of cool, seems like. But it does not work, as it looks like the students and teachers adopt to each other.
I ran it over night, and ended up with 700+ versions of students for both zeros and crosses.
However the version 700 of zero is consistenly loosing over version like 695 of the crosses, and version 1 as well.

As I am writing this, I realized that crosses, who start as a "classifier" model, keep being trained as classifier, not a player.
That may explain part of it :(
Let's try to fix it next and see what happens.

Next idea will be to compete a student with EACH of previous teachers, and generate training batch using all of them who win.
Will take a while, but seems promising.

```
python train_by_playing.py --save_to_model models/versioned/model
python generate_games.py --mode play_many_games \
    --crosses_model classifier:models/model_victory_only.json \
    --zeroes_model player:models/versioned/model-zeroes-4.4.json
```


# 2025-04-23
Added a competition into training, and stop when the student model starts to win.
It happens roughly after 10th epoch.
And it wins indeed, though I haven't checked how the game looks.
Next need to swap student and teacher and see if the progress goes on.

```
python train_by_playing.py --save_to_model models/model_barely_winning_zeroes.json
python generate_games.py --mode play_many_games --crosses_model classifier:models/model_victory_only.json --zeroes_model player:models/model_barely_winning_zeroes.json

```

# 2025-04-21
Wow is not really a wow if you watch the games closely.
They all similar, very repetitive and quite useless.
The reason is that the model found imperfection of classifier based model and just started to put zeroes in a row
blocking the crosses and building 4-line first :)
If you retrain the classifier and try to run player against it, it will lose.
Useful insight for next step - the student must becomde a teacher at some point. To be continued.

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

# Went with the feedback loop idea and wow:

python train_by_playing.py --save_to_model models/model_playing_mce_loss.json 

python generate_games.py --mode play_many_games --crosses_model classifier:models/model_victory_only.json --zeroes_model player:models/model_playing_mce_loss.json
Crosses: 13 out of 100
Zeroes: 87 out of 100
Ties: 0 out of 100

python generate_games.py --mode play_many_games --zeroes_model classifier:models/model_victory_only.json --crosses_model player:models/model_playing_mce_loss.json
Crosses: 56 out of 100
Zeroes: 44 out of 100
Ties: 0 out of 100

# It is trained to play zeroes only, but even for crosses significant progress! 
# Moving forward with that. Next we need to include "gaming competition" into the training, and as soon as student model starts to consistently win
# over the teacher, it should become a teacher itself. This can lead to some other side effects ("echo chamber") so let's see

----


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
`


# To auto indent
pip install black
black .``
