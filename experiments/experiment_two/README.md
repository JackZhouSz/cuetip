# Instructions to Replicate Experiments 4.3.2: Explanation Relevance to Expert Rules:

This experiment evaluates the grounding of the system using Llama models (70B, 8B, or 3B) either with DSPy's BootstrapFewShotWithRandomSearch (BFSwRS) for prompt optimization or with no prompt optimization (None).

## Setup Environment Variables

To set up your environment, follow these steps:
* Modify `secret.sh` with your wandb login key.
* Modify `secret.sh` with your `HUGGINGFACE_TOKEN` API key.
* Modify `secret.sh` with your OpenAI `API_KEY`.
* Run the following command: `source secret.sh`

## Generate W&B DSPy Dataset

Generate the dataset with train/validation/test splits:
* **With $\mathbf r_r$**: Use the script: `run_make_DEF_explanations_wDEF_DSPy_dataset.sh`
* **Without $\mathbf r_r$**: Use the script: `run_make_DEF_explanations_woDEF_DSPy_dataset.sh`

## Run Experiments

Execute the appropriate scripts to run the evaluations. The scripts follow the pattern:
`run_eval_DEF_explanations_[wo|w]DEF_DSPy_optim_[BFSwRS|none]_[3B|8B|70B]_SEED10-100.sh`

For example:
* To run an evaluation with $\mathbf r_r$, without prompt optimization and using Llama-3.2-8B-Instruct: `./run_eval_DEF_explanations_wDEF_DSPy_optim_none_8B_SEED10-100.sh`.
* Or, similarly with prompt optimization using DSPy's `BootstrapFewShotWithRandomSearch`: `./run_eval_DEF_explanations_wDEF_DSPy_optim_BFSwRS_8B_SEED10-100.sh`.

## Generate Results Visualizations

To get the figures for the results, and the outcomes of the statistical significance test:
* **Script**: Use `make_figure_DSPy_DEF_[BFSwRS_n4|None]_llama-3.[1-70B|1-8B|2-3B]-Instruct.sh`.
* **Config file**: As details in the script, a config file is used to fetch the relevant results from specific experiment runs: `exp2-DSPy-DEF-Llama-3.[1-70B|1-8B|2-3B]-Instruct-PP-[BFSwRS-n=4|None]-value+difficulty.json`. An example of a config file structure for the `plot_DEF_scores.py` Python script called by the figure-generating script is provided:
    ```json
    {
    "With Heuristics (BFSwRS-n=4)": [
            "./results-values-ExplainDEFWithDEF-v4-percentages-SEED10-BFSwRS-n=4-CP10-1mL1d6-llama-3.1-70B-Instruct.json"
            "./results-values-ExplainDEFWithDEF-v4-percentages-SEED20-BFSwRS-n=4-CP10-1mL1d6-llama-3.1-70B-Instruct.json"
    ],

    "Baseline (no optim.)": [
            "./results-values-ExplainDEFWithOutDEF-SEED10-None-1mL1d6-llama-3.1-70B-Instruct.json",
            "./results-values-ExplainDEFWithOutDEF-SEED20-None-1mL1d6-llama-3.1-70B-Instruct.json",
            "./results-values-ExplainDEFWithOutDEF-SEED30-None-1mL1d6-llama-3.1-70B-Instruct.json",
            "./results-values-ExplainDEFWithOutDEF-SEED40-None-1mL1d6-llama-3.1-70B-Instruct.json"
    ]
    }
    ```
* **Output visualization**: The figures will be saved as `exp2-DSPy-DEF-Llama-3.[1-70B|1-8B|2-3B]-Instruct-PP-[BFSwRS-n=4|None]-value+difficulty-AggL1Dist.pdf`, as specified in the figure-generating script. 

