#!/bin/bash

python main.py rag.pkg='numpy' rag.num_tokens=15 rag.beam_width=10 rag.epoch_num=50 rag.top_k_tokens=5
# wait
# python main.py rag.pkg='pandas' rag.num_tokens=15 rag.beam_width=10 rag.epoch_num=50 rag.top_k_tokens=5
# wait
# python main.py rag.pkg='numpy' rag.num_tokens=15 rag.beam_width=10 rag.epoch_num=50 rag.top_k_tokens=5
