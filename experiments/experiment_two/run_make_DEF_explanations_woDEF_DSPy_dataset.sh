#/bin/bash

HEADLESS=true python make_DEF_explanations_DSPy_dataset.py \
--json_path="../../data/stochastic_training_data.json" \
--value_rules_path="../../data/value_rules.json" \
--difficulty_rules_path="../../data/difficulty_rules.json" \
--with_def_signature=False \
--percentage_values=True \
--num_train_examples=200 \
--num_val_examples=100 \
--num_test_examples=100 \
--reset=True 


