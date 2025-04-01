#!/bin/bash


python test.py --pkg 'matplotlib' --num_tokens 20 --beam_width 10 --epoch_num 50 --top_k_tokens 5
wait
python test.py --pkg 'matplotlib' --num_tokens 15 --beam_width 10 --epoch_num 200 --top_k_tokens 5
