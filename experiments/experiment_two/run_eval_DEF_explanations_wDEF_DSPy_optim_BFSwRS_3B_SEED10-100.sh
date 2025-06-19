#/bin/bash

if [ -z "$HUGGINGFACE_TOKEN" ]; then
    echo "Error: HUGGINGFACE_TOKEN is not set"
    exit 1
fi

python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.2-3B-Instruct &
sleep 120

python eval_DEF_explanations_DSPy_optim.py \
--seed=10 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED10-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=20 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED20-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=30 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED30-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=40 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED40-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=50 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED50-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=60 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED60-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=70 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED70-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=80 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED80-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=90 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED90-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

python eval_DEF_explanations_DSPy_optim.py \
--seed=100 \
--n_threads=4 \
--task_name="explain_def_task_with_def_percentages" \
--project_name="SimLM-DEF-explanations" \
--module_path="./ExplainDEFWithDEF-v4-percentages-SEED100-BFSwRS-n=4-CP10-1mL1d6-llama-3.2-3B-Instruct.json" \
--model_id="meta-llama/Llama-3.2-3B-Instruct" \
--port=7501 \
--temperature=0.7 \
--max_tokens=4096 \
--n_samples=3 \
--nbr_demos=4 \
--teleprompter_type="bootstrap" \
--with_def_signature=True \
--reset=True

pkill -f "python -m sglang.launch_server --port 7501 --model-path meta-llama/Llama-3.2-3B-Instruct"

