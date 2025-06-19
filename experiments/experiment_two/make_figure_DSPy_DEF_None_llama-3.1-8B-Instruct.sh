#/bin/bash

python plot_DEF_scores.py \
--dimension=0 \
--rule_type=difficulty \
--json_config=./exp2-DSPy-DEF-Llama-3.1-8B-Instruct-PP-None-value+difficulty.json \
--output_filepath=./exp2-DSPy-DEF-Llama-3.1-8B-Instruct-PP-None-value+difficulty \
--format=pdf \
--show_pvalues=False \
--alpha=0.8 \
--figaspect=0.18 \
--figwidth=28.24 \
--font_size=9 \
--font_scale=4.0

