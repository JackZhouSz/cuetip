#/bin/bash

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi

python -m sglang.launch_server --tp 2 --port 7502 --model-path meta-llama/Llama-3.1-70B-Instruct --sampling-backend=pytorch &
sleep 1200

python eval_DEF_explanations_DSPy_optim.py \
--seed=10 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED10-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=20 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED20-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=30 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED30-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=40 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED40-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=50 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED50-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=60 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED60-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=70 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED70-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=80 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED80-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=90 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED90-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=100 \
--n_threads=4 \
--task_name="explain_def_task_without_def" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithOutDEF-SEED100-None-1mL1d6-llama-3.1-70B-Instruct.json" \
--model_id="meta-llama/Llama-3.1-70B-Instruct" \
--port=7502 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="none" \
--with_def_signature=False \
--reset=True

pkill -f "python -m sglang.launch_server --tp 2 --port 7502 --model-path meta-llama/Llama-3.1-70B-Instruct"

