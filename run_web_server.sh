#python3 -m web.server --zeroes_model player:models/fixed_rounds/model-zeroes-956.json
#python3 -m web.server --zeroes_model player:models/shorter_rounds/model-zeroes-1000.json
#python3 -m web.server --zeroes_model player:models/shorter_rounds/model-zeroes-1132.json

# 1440 seems to work better than above, but also better than 3000 below. Interesting
#python3 -m web.server --zeroes_model player:models/try_again/model-zeroes-1440.json
#python3 -m web.server --zeroes_model player:models/try_again/model-zeroes-2400.json
#python3 -m web.server --zeroes_model player:models/try_again/model-zeroes-3000.json
python3 -m web.server --crosses_model player:models/cpp/model-crosses-13000.json --zeroes_model player:models/cpp/model-zeroes-3000.json
