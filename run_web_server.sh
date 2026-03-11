#python3 -m web.server --model_o player:models/fixed_rounds/model-zeroes-956.json
#python3 -m web.server --model_o player:models/shorter_rounds/model-zeroes-1000.json
#python3 -m web.server --model_o player:models/shorter_rounds/model-zeroes-1132.json

# 1440 seems to work better than above, but also better than 3000 below. Interesting
#python3 -m web.server --model_o player:models/try_again/model-zeroes-1440.json
#python3 -m web.server --model_o player:models/try_again/model-zeroes-2400.json
#python3 -m web.server --model_x player:models/try_again/model-crosses-3000.json --model_o player:models/try_again/model-zeroes-3000.json
#python3 -m web.server --model_x player:models/cpp/model-crosses-13000.json --model_o player:models/cpp/model-zeroes-3000.json
#python3 -m web.server --model_x player:models/cpp3/model-crosses-d.3300.json --model_o player:models/cpp3/model-zeroes-a.2800.json
python3 -m web.server --model_x player:models/cpp3.001/duomodel-4000.json  --model_o player:models/cpp3.001/duomodel-4000.json
