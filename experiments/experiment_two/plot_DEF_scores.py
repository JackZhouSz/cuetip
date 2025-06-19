import json
import copy
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import argparse
import numpy as np
import random
import os
from typing import List, Dict, Tuple
from enum import Enum
import scipy.stats as stats


class EstimateEnum(Enum):
    very_low = "very low"
    low = "low"
    moderately_low = "moderately low"
    moderate = "moderate"
    moderately_high = "moderately high"
    high = "high"
    very_high = "very high"
    unknown = "UNKNOWN"

# Map ranges of [-1, 1] to EstimateEnum values
def map_to_enum(value: float) -> EstimateEnum:
    bins = np.linspace(-1, 1, len(EstimateEnum) - 1)  # Exclude "unknown"
    enums = list(EstimateEnum)[:-1]  # Exclude "unknown"
    for i, threshold in enumerate(bins):
        if value <= threshold:
            return enums[i]
    return enums[-1]

# Extract EstimateEnum from sentences
def extract_enum(sentence: str) -> EstimateEnum:
    for e in EstimateEnum:
        if e.value in sentence:
            return e
    return EstimateEnum.unknown


def compute_accuracy_v0(data: List[Dict]) -> List[Dict]:
    """
    Compute accuracy based on the most-voted EstimateEnum values for each rule (dimension).
    Args:
        data (List[Dict]): List of data points with estimations and ground truth weights.
    Returns:
        List[Dict]: Processed list of data points with accuracy metrics for each rule.
    """
    processed_data = []

    for datapoint in data:
        # Convert ground truth weights to enums
        groundtruth_enums = [
            map_to_enum(weight)
            for weight in datapoint['estimations']["groundtruth_weights"]["value_weights"]
        ]

        metrics_per_rule = [{} for _ in groundtruth_enums]  # Create a structure for each rule

        for category in ["with_functions", "without_functions"]: #TODO, "random"]:
            if datapoint['estimations'][category]:
                # Extract enums from the list of sentences for each rule
                enum_values_per_rule = [
                    [extract_enum(sentence) for sentence in est["value_explanations"]]
                    for est in datapoint['estimations'][category]
                ]

                # Transpose to organize by rule
                import ipdb; ipdb.set_trace()
                enums_by_rule = list(zip(*enum_values_per_rule))

                # Calculate most-voted enums and metrics per rule
                for rule_index, (groundtruth, rule_enums) in enumerate(zip(groundtruth_enums, enums_by_rule)):
                    enum_counts = pd.Series(rule_enums).value_counts()
                    most_voted = enum_counts.idxmax() if not enum_counts.empty else EstimateEnum.unknown

                    total_count = len(rule_enums)
                    correct_count = sum(1 for pred in rule_enums if pred == groundtruth)
                    unknown_count = rule_enums.count(EstimateEnum.unknown)

                    metrics_per_rule[rule_index][category] = {
                        "accuracy": correct_count / total_count if total_count else 0,
                        "unknown_percentage": unknown_count / total_count if total_count else 0,
                        "inaccuracy": 1 - (correct_count + unknown_count) / total_count if total_count else 0,
                    }

        processed_data.append(metrics_per_rule)

    return processed_data

def create_stacked_barplot_v0(
    data: List[Dict],
    dimension_index: int,
    output_filepath: str,
    figsize: Tuple[int] = (10, 6),
):
    """
    Create a stacked bar plot for accuracy, unknown, and inaccuracy percentages for a specific rule.
    Args:
        data (List[Dict]): List of accuracy metrics for each rule.
        dimension_index (int): Index of the rule (dimension) to visualize.
        output_filepath (str): Filepath to save the resulting figure.
    """
    categories = ["with_functions", "without_functions", "random"]
    aggregated_data = []

    for datapoint in data:
        if dimension_index < len(datapoint):  # Ensure the rule index is valid
            metrics = datapoint[dimension_index]
            for category in categories:
                if category in metrics:
                    aggregated_data.append({
                        "Method": category.replace("_", " ").title(),
                        "Accuracy": metrics[category]["accuracy"] * 100,
                        "Unknown": metrics[category]["unknown_percentage"] * 100,
                        "Inaccuracy": metrics[category]["inaccuracy"] * 100,
                    })

    df = pd.DataFrame(aggregated_data)

    # Create stacked barplot
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df, x="Method", y="Accuracy", color="green", label="Accuracy", ax=ax)
    sns.barplot(data=df, x="Method", y="Unknown", color="blue", label="Unknown", ax=ax, bottom=df["Accuracy"])
    sns.barplot(data=df, x="Method", y="Inaccuracy", color="red", label="Inaccuracy", ax=ax, bottom=df["Accuracy"] + df["Unknown"])

    ax.set_ylabel("Percentage")
    ax.set_xlabel("")
    ax.set_title(f"Accuracy Metrics for Rule {dimension_index + 1}")
    ax.legend(title="Metric")
    plt.tight_layout()

    plt.savefig(output_filepath)
    print(f"Figure saved to {output_filepath}")
    plt.close()


def compute_accuracy(data: List[Dict]) -> List[Dict]:
    """
    Compute accuracy based on the most-voted EstimateEnum values for each rule across datapoints.
    Args:
        data (List[Dict]): List of data points with estimations and ground truth weights.
    Returns:
        List[Dict]: Processed list of accuracy metrics for each rule.
    """
    # Extract number of rules from ground truth
    num_rules = len(data[0]['estimations']["groundtruth_weights"]["value_weights"])
    metrics_per_rule = [{} for _ in range(num_rules)]

    for rule_index in range(num_rules):
        rule_metrics = {"correct": 0, "unknown": 0, "total": 0}

        for datapoint in data:
            # Convert the ground truth for this rule to an enum
            groundtruth_enum = map_to_enum(datapoint['estimations']["groundtruth_weights"]["value_weights"][rule_index])

            most_voted_per_category = {}

            for category in ["with_functions", "without_functions"]: #TODO, "random"]:
                if datapoint['estimations'][category]:
                    # Collect enums for this rule across all samples
                    rule_enums = [
                        extract_enum(est["value_explanations"][rule_index])
                        for est in datapoint['estimations'][category]
                    ]

                    # Determine the most-voted enum
                    enum_counts = pd.Series(rule_enums).value_counts()
                    most_voted = enum_counts.idxmax() if not enum_counts.empty else EstimateEnum.unknown
                    most_voted_per_category[category] = most_voted

            # For each category, update the rule-level metrics
            for category, most_voted in most_voted_per_category.items():
                if category not in metrics_per_rule[rule_index]:
                    metrics_per_rule[rule_index][category] = {"correct": 0, "unknown": 0, "total": 0}

                rule_metrics = metrics_per_rule[rule_index][category]
                rule_metrics["total"] += 1
                if most_voted == EstimateEnum.unknown:
                    rule_metrics["unknown"] += 1
                elif most_voted == groundtruth_enum:
                    rule_metrics["correct"] += 1

    # Compute accuracy and percentages for each rule and category
    for rule_metrics in metrics_per_rule:
        for category, metrics in rule_metrics.items():
            total = metrics["total"]
            metrics["accuracy"] = metrics["correct"] / total if total else 0
            metrics["unknown_percentage"] = metrics["unknown"] / total if total else 0
            metrics["inaccuracy"] = 1 - (metrics["correct"] + metrics["unknown"]) / total if total else 0

    return metrics_per_rule


def create_stacked_barplot(
    data: List[Dict],
    dimension_index: int,
    rule_type: str,
    output_filepath: str,
    figsize: Tuple[int] = (10, 6),
):
    """
    Create a stacked bar plot for accuracy, unknown, and inaccuracy percentages for a specific rule.
    Args:
        data (List[Dict]): List of accuracy metrics for each rule.
        dimension_index (int): Index of the rule (dimension) to visualize.
        output_filepath (str): Filepath to save the resulting figure.
    """
    rule_metrics = data[dimension_index]
    aggregated_data = []

    category2legend = {
        'with_functions': 'Ours',
        'without_functions': 'Baseline',
    }

    for category, metrics in rule_metrics.items():
        aggregated_data.append({
            "Method": category2legend[category], #category.replace("_", " ").title(),
            "Accuracy": metrics["accuracy"] * 100,
            "Unknown": metrics["unknown_percentage"] * 100,
            "Inaccuracy": metrics["inaccuracy"] * 100,
        })

    df = pd.DataFrame(aggregated_data)

    # Create stacked barplot
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    fig, ax = plt.subplots(figsize=figsize)

    sns.barplot(data=df, x="Method", y="Accuracy", color="green", label="Accurate", ax=ax)
    sns.barplot(data=df, x="Method", y="Unknown", color="blue", label="Failure-to-Answer", ax=ax, bottom=df["Accuracy"])
    sns.barplot(data=df, x="Method", y="Inaccuracy", color="red", label="Inaccurate", ax=ax, bottom=df["Accuracy"] + df["Unknown"])

    ax.set_ylabel("Proportion (%)")
    ax.set_xlabel("")
    ax.set_title(f"Multiple-choice QA for {rule_type.title()} DEF {dimension_index + 1}")
    ax.legend(title="Answers")
    plt.tight_layout()

    output_filepath += f'-R{dimension_index+1}.png'
    plt.savefig(output_filepath)
    print(f"Figure saved to {output_filepath}")
    plt.close()


def str2bool(inp):
    if not isinstance(inp, bool):
        assert isinstance(inp, str)
        inp = 'true' in inp.lower()
    return inp



# Map EstimateEnum to numerical values
enum_to_numeric = {
    EstimateEnum.very_low: 0,
    EstimateEnum.low: 1,
    EstimateEnum.moderately_low: 2,
    EstimateEnum.moderate: 3,
    EstimateEnum.moderately_high: 4,
    EstimateEnum.high: 5,
    EstimateEnum.very_high: 6,
    EstimateEnum.unknown: None,  # Exclude 'unknown' from distance calculations
}

def compute_accuracy_and_l1(
    data: List[Dict],
    rule_type: str,
) -> List[Dict]:
    """
    Compute accuracy, unknown percentages, inaccuracy, and average L1 distance for each rule.
    Args:
        data (List[Dict]): List of data points with estimations and ground truth weights.
        rule_type (str): rule type to consider, among {value, difficulty}.
    Returns:
        List[Dict]: Processed list of metrics, including L1 distances for each rule.
    """
    max_distance = max([v for v in enum_to_numeric.values() if isinstance(v,int)])
    est_entry = 'estimations'
    if 'estimations' not in list(data[0].keys()):
        est_entry = 'explanations'
    if 'groundtruth_weights' not in data[0][est_entry]:
        # Need to extract the groundtruth values from f'normalized_{rule_type}'
        # And, possibly, those may be represented as percentages, so it is necessary to 
        # put them back into [-1,+1] range:
        for datapoint in data:
            if 'value' in rule_type:
                norm_w = datapoint[f'normalized_values']
            else:
                norm_w = datapoint[f'normalized_difficulty']
            
            # processing:
            if max(norm_w) > 1.0:
                #percentages:
                p_norm_w = [w/100.0*2-1 for w in norm_w]
            else:
                #scalars:
                p_norm_w = [w for w in norm_w]
            datapoint[est_entry]['groundtruth_weights'] = {
                f"{rule_type}_weights": p_norm_w,
            }

    num_rules = len(data[0][est_entry]["groundtruth_weights"][f"{rule_type}_weights"])
    metrics_per_rule = [{} for _ in range(num_rules)]

    for rule_index in range(num_rules):
        for category in ["with_functions", "without_functions"]:
            rule_metrics = {"correct": 0, "unknown": 0, "total": 0, "l1_distances": []}

            for datapoint in data:
                # Convert ground truth to numerical value
                groundtruth_enum = map_to_enum(datapoint[est_entry]["groundtruth_weights"][f"{rule_type}_weights"][rule_index])
                groundtruth_numeric = enum_to_numeric.get(groundtruth_enum)

                if datapoint[est_entry][category]:
                    # Collect enums for this rule across all samples
                    rule_enums = [
                        extract_enum(est[f"{rule_type}_explanations"][rule_index])
                        for est in datapoint[est_entry][category]
                    ]

                    # Determine the most-voted enum
                    enum_counts = pd.Series(rule_enums).value_counts()
                    most_voted = enum_counts.idxmax() if not enum_counts.empty else EstimateEnum.unknown
                    most_voted_numeric = enum_to_numeric.get(most_voted)

                    # Update metrics
                    rule_metrics["total"] += 1
                    if most_voted == EstimateEnum.unknown:
                        rule_metrics["unknown"] += 1
                    elif most_voted == groundtruth_enum:
                        rule_metrics["correct"] += 1

                    # Calculate L1 distance if valid
                    if groundtruth_numeric is not None and most_voted_numeric is not None:
                        rule_metrics["l1_distances"].append(abs(groundtruth_numeric - most_voted_numeric))
                    elif most_voted_numeric is None:
                        rule_metrics["l1_distances"].append(max_distance)

            # Aggregate metrics for the category
            if rule_metrics["total"] > 0:
                rule_metrics["accuracy"] = rule_metrics["correct"] / rule_metrics["total"]
                rule_metrics["unknown_percentage"] = rule_metrics["unknown"] / rule_metrics["total"]
                rule_metrics["inaccuracy"] = 1 - (rule_metrics["correct"] + rule_metrics["unknown"]) / rule_metrics["total"]
                rule_metrics["average_l1_distance"] = np.mean(rule_metrics["l1_distances"]) if rule_metrics["l1_distances"] else None

            metrics_per_rule[rule_index][category] = rule_metrics

    return metrics_per_rule

def create_stacked_barplot_with_l1(
    data: List[Dict],
    dimension_index: int,
    rule_type: str,
    output_filepath: str,
    figsize: Tuple[int] = (10, 6),
):
    """
    Create a stacked bar plot for accuracy, unknown, and inaccuracy percentages, with L1 distance.
    Args:
        data (List[Dict]): List of metrics for each rule, including L1 distances.
        dimension_index (int): Index of the rule (dimension) to visualize.
        output_filepath (str): Filepath to save the resulting figure.
    """
    rule_metrics = data[dimension_index]
    aggregated_data = []
    l1_distances = []

    for category, metrics in rule_metrics.items():
        aggregated_data.append({
            "Method": category.replace("_", " ").title(),
            "Accuracy": metrics["accuracy"] * 100 if metrics["accuracy"] is not None else 0,
            "Unknown": metrics["unknown_percentage"] * 100 if metrics["unknown_percentage"] is not None else 0,
            "Inaccuracy": metrics["inaccuracy"] * 100 if metrics["inaccuracy"] is not None else 0,
        })
        l1_distances.append({
            "Method": category.replace("_", " ").title(),
            "L1 Distance": metrics["average_l1_distance"] if metrics["average_l1_distance"] is not None else 0,
        })

    df = pd.DataFrame(aggregated_data)
    df_l1 = pd.DataFrame(l1_distances)

    # Create stacked barplot
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    fig, ax1 = plt.subplots(figsize=figsize)

    sns.barplot(data=df, x="Method", y="Accuracy", color="green", label="Accuracy", ax=ax1)
    sns.barplot(data=df, x="Method", y="Unknown", color="blue", label="Unknown", ax=ax1, bottom=df["Accuracy"])
    sns.barplot(data=df, x="Method", y="Inaccuracy", color="red", label="Inaccuracy", ax=ax1, bottom=df["Accuracy"] + df["Unknown"])

    # Add L1 distance as another bar plot
    ax2 = ax1.twinx()
    sns.lineplot(
        data=df_l1, x="Method", y="L1 Distance", color="purple", marker="o", label="L1 Distance", ax=ax2
    )
    ax2.set_ylabel("Average L1 Distance", color="purple")
    ax2.tick_params(axis='y', labelcolor="purple")

    ax1.set_ylabel("Percentage")
    ax1.set_xlabel("")
    ax1.set_title(f"Metrics and L1 Distance for Rule {dimension_index + 1}")
    ax1.legend(title="Metric")
    plt.tight_layout()

    plt.savefig(output_filepath)
    print(f"Figure saved to {output_filepath}")
    plt.close()


def create_l1_distance_boxplot(
    data: List[Dict],
    dimension_index: int,
    output_filepath: str,
    figsize: Tuple[int] = (10, 6),
):
    """
    Create a boxplot for the L1 distances of a specific rule across all categories.
    Args:
        data (List[Dict]): List of metrics for each rule, including L1 distances.
        dimension_index (int): Index of the rule (dimension) to visualize.
        output_filepath (str): Filepath to save the resulting figure.
    """
    rule_metrics = data[dimension_index]
    l1_data = []

    # Gather L1 distances for the specified rule
    for category, metrics in rule_metrics.items():
        if "l1_distances" in metrics:
            for l1_value in metrics["l1_distances"]:
                l1_data.append({
                    "Method": category.replace("_", " ").title(),
                    "L1 Distance": l1_value,
                })

    df = pd.DataFrame(l1_data)

    # Create boxplot
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    fig, ax = plt.subplots(figsize=figsize)

    sns.boxplot(
        data=df, 
        x="Method", 
        y="L1 Distance", 
        #palette="muted", 
        hue="Method",
        ax=ax,
        legend=False,
    )
    sns.stripplot(
        data=df, x="Method", y="L1 Distance", color="black", alpha=0.6, jitter=True, ax=ax
    )

    ax.set_ylabel("L1 Distance")
    ax.set_xlabel("")
    ax.set_title(f"L1 Distance Distribution for Value DEF {dimension_index + 1}")
    plt.tight_layout()

    output_filepath += f'-R{dimension_index+1}.png'
    plt.savefig(output_filepath)
    print(f"Figure saved to {output_filepath}")
    plt.close()


def create_aggregated_l1_boxplot_with_mean(
    data: List[Dict],
    rule_type: str,
    output_filepath: str,
    figsize: Tuple[int] = (14, 8),
    args = None,
    **kwargs,
):
    """
    Create an aggregated boxplot for L1 distances across all rules and methods,
    with mean values shown on the plot.
    Args:
        data (List[Dict]): List of metrics for all rules, including L1 distances.
        output_filepath (str): Filepath to save the resulting figure.
    """
    l1_data = []
    
    if kwargs['json_config'] is None:
        category2legend = {
            'with_functions': 'With Heuristics',
            'without_functions': 'Baseline',
        }
        # Gather L1 distances for all rules and methods
        for rule_index, rule_metrics in enumerate(data):
            for category, metrics in rule_metrics.items():
                if "l1_distances" in metrics:
                    for l1_value in metrics["l1_distances"]:
                        l1_data.append({
                            "Rule ID": f"Rule {rule_index + 1}",
                            #"Method": category.replace("_", " ").title(),
                            "Method": category2legend[category],
                            "L1 Distance": l1_value,
                        })
    else:
        # Load yaml config:
        with open(kwargs['json_config'], 'r') as f:
            config = json.load(f)
        legend2Lresults = config
        
        # Load results:
        results = {}
        for legend, results_paths in legend2Lresults.items():
            results[legend] = {}
            for path in results_paths:
                if isinstance(path, list):
                    # If multiple paths have been provided,
                    # then we need to merge the results by 
                    # concatenating on the rule_idx axis :
                    offset = 0
                    init = False
                    for pidx, strpath in enumerate(path):
                        assert isinstance(strpath, str)
                        try:
                            with open(strpath, 'r') as f:
                                tmpresults = json.load(f)
                        except Exception as e:
                            print(f"Loading of {strpath} failed:\n {e}")
                            continue
                        # Init:
                        if not init : 
                            init = True
                            results[legend][str(path)] = tmpresults
                            maxruleidx = max([int(ridx) for ridx in tmpresults.keys()])
                        else:
                            # Update:
                            maxruleidx = 0
                            for rule_idx, data_d in tmpresults.items():
                                maxruleidx = max(maxruleidx, int(rule_idx))
                                offsetrule_idx = int(rule_idx)+offset
                                results[legend][str(path)][offsetrule_idx] = data_d
                        # Regularise offset:
                        offset += maxruleidx
                    if not init:
                        continue
                else:
                    assert isinstance(path, str)
                    with open(path, 'r') as f:
                        results[legend][path] = json.load(f)
                for rule_idx, data_d  in results[legend][str(path)].items():
                    l1_values = list(data_d.values())[0]['l1_distances']
                    l1_value = np.mean(l1_values)
                    l1_data.append({
                        #"Rule ID": f"Rule {rule_idx}",
                        "Rule ID": f"{rule_idx}",
                        "Method": legend,
                        "L1 Distance": l1_value,
                    })
        #
    df = pd.DataFrame(l1_data)

    # Calculate means for overlay
    mean_df = df.groupby(["Rule ID", "Method"])["L1 Distance"].mean().reset_index()
    #summary_df = df.groupby("Method")["L1 Distance"].agg(["mean", "sem"]).reset_index()
    #summary_df = df.groupby("Method").agg(mean=("L1 Distance", "mean"), sem=("L1 Distance", "sem")).reset_index()

    # Ensure there are exactly two methods for the KS test
    methods = df["Method"].unique()
    print(methods)
    if len(methods) != 2:
        #raise ValueError("The KS test requires exactly two methods.")
        print(f"WARNING: The KS test requires exactly two methods.")
        print(f"WARNING: KS will be computed over the first 2 methods:")
        halfidx = int(len(methods)/2)
        print(methods[0], methods[halfidx])
    else:
        halfidx = 1

    # Perform KS test for each rule
    ks_p_values = []
    for rule_id in df["Rule ID"].unique():
        # test that baseline is lower than heuristic (alternative, meaning hoping for small p-value)
        type1_data = df[(df["Rule ID"] == rule_id) & (df["Method"] == methods[0])]["L1 Distance"]
        type2_data = df[(df["Rule ID"] == rule_id) & (df["Method"] == methods[halfidx])]["L1 Distance"]
        _, p_value = stats.ks_2samp(
            type1_data, 
            type2_data,
            alternative="greater",#"less",
            method="auto",
        )
        ks_p_values.append(p_value.item())
    
    print(f"P-values: {ks_p_values}")

    mtype1_data = df[df["Method"] == methods[0]]["L1 Distance"]
    mtype2_data = df[df["Method"] == methods[halfidx]]["L1 Distance"]
    
    mtype1_data_mean = np.mean(mtype1_data)
    mtype1_data_std = np.std(mtype1_data)

    mtype2_data_mean = np.mean(mtype2_data)
    mtype2_data_std = np.std(mtype2_data)

    print(f"{methods[0]} mean+/-std: {mtype1_data_mean} +/- {mtype1_data_std}") 
    print(f"{methods[halfidx]} mean+/-std: {mtype2_data_mean} +/- {mtype2_data_std}") 

    _, mean_p_value = stats.ks_2samp(
        mtype1_data, 
        mtype2_data,
        alternative="greater", #"less",
        method="auto",
    )
    mean_ks_p_values = [mean_p_value]
    print(mean_p_value)

    # Define custom color palette
    #custom_colors = ['#d15d08', '#0474b4', '#c783a6']
    custom_colors = ['#d15d08', '#0474b4', '#e26e00', '#158500', ]
    method_colors = dict(zip(df["Method"].unique(), custom_colors))

    # Create aggregated boxplot
    sns.set(style="whitegrid")
    sns.set_context("notebook", font_scale=args.font_scale)
    #fig, ax = plt.subplots(figsize=figsize)
    #fig, (ax, axr) = plt.subplots(1, 2, figsize=figsize, gridspec_kw={"width_ratios": [4, 1]})

    '''
    font = {
        #'family' : 'normal',
        'family' : 'serif',
        'serif': ['Helvetica Neue'],
        #'weight' : 'bold',
        'size'   : 9,
    }

    matplotlib.rc('font', **font)
    '''
    
    rc_fonts = {
    "font.family": "serif",
    "font.size": args.font_size,
    'figure.figsize': (5, 3),
    "text.usetex": True,
    #'text.latex.preview': True,
    #'text.latex.preamble': [
    #    r"\usepackage[libertine]{newtxmath}",
    #    r"\usepackage{libertine}",
    #    ],
    #'text.latex.preamble': r"\usepackage[libertine]{newtxmath}",
    'text.latex.preamble': r"\usepackage{libertine}",
    # Embedding the font in the PDF:
    "pdf.fonttype": 42,
    }
    matplotlib.rcParams.update(rc_fonts)

    #fig = plt.figure(figsize=figsize)
    fig = plt.figure(figsize=matplotlib.figure.figaspect(args.figaspect))
    fig.set_size_inches(args.figwidth, args.figwidth*args.figaspect)
    #fig.set_dpi(1800)
    if args.show_pvalues:
        gs = fig.add_gridspec(
            2, 2, 
            height_ratios=[3, 0.2], 
            #height_ratios=[3, 0.5], 
            #width_ratios=[4, 1], 
            width_ratios=[14, 1], 
            #hspace=0.05, 
            hspace=0.1, 
            wspace=0.2,
        )
    else:
        gs = fig.add_gridspec(
            1, 2, 
            #height_ratios=[3, 0.2], 
            #height_ratios=[3, 0.5], 
            #width_ratios=[4, 1], 
            width_ratios=[28, 1], 
            #hspace=0.05, 
            #hspace=0.1, 
            #wspace=0.2,
        )

    # Left: Aggregated boxplot with means
    ax = fig.add_subplot(gs[0, 0])
    
    # sns.boxplot(data=df, x="Rule ID", y="L1 Distance", hue="Method", palette="muted", ax=ax)
    # sns.stripplot(
    #     data=df, x="Rule ID", y="L1 Distance", hue="Method", 
    #     color="black", alpha=0.6, jitter=True, dodge=True, ax=ax
    # )
    # 
    # 
    # Overlay means as points
    # sns.lineplot(
    #     data=df, #mean_df, 
    #     x="Rule ID", 
    #     y="L1 Distance", 
    #     hue="Method",
    #     color="purple", 
    #     markers=["o","D","s"], 
    #     #label="L1 Distance",
    #     ax=ax,
    # )
    
    sns.barplot(
        data=df, 
        x="Rule ID", 
        y="L1 Distance", 
        hue="Method",
        errorbar="se", 
        palette=method_colors,#"dark", 
        alpha=args.alpha,
        #capsize=.6,
        err_kws={'linewidth': 2.5},
        #err_kws={'linewidth': 5},
        #err_kws={'linewidth': 2.5, }, #'linestyle':'-', 'antialiased': True},
        #err_kws={"color": ".5", "linewidth": 2.5},
        ax=ax,
    )
    # sns.scatterplot(
    #     data=mean_df, x="Rule ID", y="L1 Distance", hue="Method",
    #     style="Method", markers=["o", "s", "D"], 
    #     #palette="dark", 
    #     #edgecolor="black",
    #     palette='dark:black',
    #     s=100, ax=ax, 
    #     legend=False
    # )
    # 
    # 
    # Customize plot
    #ax.set_ylabel("L1 Distance")
    #ax.set_ylabel("Absolute Distance")
    ax.set_ylabel("Distance")
    ax.legend().set_visible(False) #remove()
    #ax.legend(loc='upper left', bbox_to_anchor=(0.095, 1.02)).set_title("")
    #ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.02)).set_title("")
    #fig.legend(loc='outside upper right').set_title("")

    if args.show_pvalues:
        ax.set_xlabel(f"")
        ax.set_xticks([])
    else:
        ax.set_xlabel(f"")
        #ax.set_xlabel(f"Rule ID")
    #ax.set_title(f"Aggregated L1 Distance Distribution Across {rule_type.title()} Rules with Means")
    #ax.set_title(f"L1 Distance Distribution Per {rule_type.title()} Rule (Mean +/- Std. Err. Accross Datapoint)")
    #ax.set_title(f"Absolute Distance Distribution Per {rule_type.title()} Rule (Mean +/- Std. Err. Accross Seeds)")
    # Place legend inside the left plot
    #ax.legend(title="Method", loc="upper left", bbox_to_anchor=(0.02, 0.98))
    #ax.legend(title="Method", bbox_to_anchor=(1.05, 1), loc="upper left")
    
    # Heatmap of KS test p-values placed below the x-axis
    if args.show_pvalues:
        ax_heatmap = fig.add_subplot(gs[1, 0])
        sns.heatmap(
            [ks_p_values],
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            cbar=False,
            xticklabels=df["Rule ID"].unique(),
            yticklabels=["P-values"],#[f"{methods[0]} < {methods[1]}"],
            ax=ax_heatmap,
        )
        #ax_heatmap.set_xticklabels(ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        ax_heatmap.set_yticklabels(ax_heatmap.get_yticklabels(), rotation=0, ha="right")
        #ax_heatmap.tick_params(axis="x", which="major", labelsize=10)
        ax_heatmap.set_xlabel("Rule ID")
        ax_heatmap.set_ylabel("")


    # Mean accross rules:
    axr = fig.add_subplot(gs[0, 1])
    sns.barplot(
        data=df,#summary_df, 
        y="L1 Distance", 
        x="Method", 
        #orient="h",
        hue="Method",
        legend=True, #False,
        palette=method_colors,#"muted",
        errorbar='se', 
        #capsize=.6,
        err_kws={'linewidth': 2.5,},# 'linestyle':'-', 'antialiased': True},
        alpha=args.alpha,
        ax=axr,
    )
    '''
    for index, row in summary_df.iterrows():
        axr.errorbar(
            x=row["mean"], 
            y=index, 
            xerr=row["sem"], 
            fmt="none", 
            c="black", 
            capsize=5
        )
    '''
    axr.set_xlabel("")#L1 Distance")
    axr.set_ylabel("")
    #if args.show_pvalues:
    axr.set_xticks([])
    #axr.set_title(f"Mean +/- Std. Err. Across {rule_type.title()} Rules")
 
    # Align y-axis limits between the two plots
    y_limits = ax.get_ylim()
    axr.set_ylim(y_limits)
    #axr.legend().set_visible(False)
    axr.legend(
        #loc='center right', 
        bbox_to_anchor=(1.02, 0.72),
        ncol=1,
        #fancybox=False,
        frameon=False,
    ).set_title("")
    '''
    axr.legend(
        loc='upper left', 
        bbox_to_anchor=(1.02, 1.02),
        ncol=1,
        #fancybox=False,
        frameon=False,
    ).set_title("")
    '''
    #fig.legend(loc="outside center right").set_title("")

    if args.show_pvalues:
        # Heatmap of KS test p-values placed below the x-axis for means
        ax_heatmapR = fig.add_subplot(gs[1, 1])
        sns.heatmap(
            [mean_ks_p_values],
            annot=True,
            fmt=".4f",
            cmap="coolwarm",
            cbar=False,
            xticklabels=['  '.join(df["Method"].unique())],
            yticklabels=["P-values"],#[f"{methods[0]} < {methods[1]}"],
            ax=ax_heatmapR,
        )
        #ax_heatmapR.set_xticklabels(ax_heatmapR.get_xticklabels(), rotation=0, ha="right")
        ax_heatmapR.set_yticklabels(ax_heatmapR.get_yticklabels(), rotation=0, ha="right")
        #ax_heatmapR.tick_params(axis="x", which="major", labelsize=10)
        ax_heatmapR.set_xlabel("Method")
        ax_heatmapR.set_ylabel("")

    plt.tight_layout()
    #plt.tight_layout(pad=0.4, w_pad=1.5, h_pad=1.0)

    # Save figure
    plt.savefig(output_filepath)
    #plt.savefig(output_filepath, bbox_inches='tight')
    print(f"Aggregated boxplot with means saved to {output_filepath}")
    plt.close()


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Compute accuracy and visualize dimension-specific data.")
    parser.add_argument("--model", type=str, help="Model to compute metric or make figures for.")
    parser.add_argument("--json_config", type=str, default=None, help="Path to the JSON file to use in plotting.. If set, then input_filepath is ignored.")
    parser.add_argument("--input_filepath", type=str, help="Path to the input JSON file.")
    parser.add_argument("--output_filepath", type=str, help="Path to save computed distances and figures.")
    parser.add_argument("--rule_type", type=str, default='value', help="Type of DEF to consider, among {value, difficulty'}.")
    parser.add_argument("--dimension", type=int, default=0, help="Index of the dimension to visualize.")
    parser.add_argument("--reset", type=str2bool, default=False, help="Compute distances from scratch even if output exists.")
    parser.add_argument("--figsize", type=str, default="10,6", help="Figure size as 'width,height' (default: '10,6').")
    parser.add_argument("--alpha", type=float, default=0.8, help="Figure element's alpha ratio.")
    parser.add_argument("--figaspect", type=float, default=0.25, help="Figure aspect ratio.")
    parser.add_argument("--figwidth", type=float, default=7.06, help="Figure width size in inches.")
    parser.add_argument("--format", type=str, default="pdf", help="Format in which to output the figure.")
    parser.add_argument("--font_size", type=float, default=9, help="Font size (pt) for figure.")
    parser.add_argument("--font_scale", type=float, default=1.2, help="Font scaling for figure.")
    parser.add_argument("--show_pvalues", type=str2bool, default=True, help="Whether to show p-values")

    args = parser.parse_args()
   
    # Parse figure size
    try:
        figsize = tuple(map(float, args.figsize.split(",")))
    except ValueError:
        raise ValueError("Invalid format for --figsize. Use 'width,height' (e.g., '10,6').")
 
    if args.json_config is None:
        # Check if the output file already exists
        if os.path.exists(args.output_filepath) and not args.reset:
            print(f"Loading precomputed accuracy from {args.output_filepath}...")
            with open(args.output_filepath, "r") as file:
                output_data = json.load(file)
        else:
            # Load the input data
            print(f"Computing accuracy from input file {args.input_filepath}...")
            with open(args.input_filepath, "r") as file:
                input_data = json.load(file)
        
            # Compute accuracy
            data = input_data[args.model]
            #output_data = compute_accuracy(data)
            output_data = compute_accuracy_and_l1(
                data,
                rule_type=args.rule_type,
            )
        
            # Save 
            with open(args.output_filepath, "w") as file:
                json.dump(output_data, file, indent=4)
            print(f"Accuracy saved to {args.output_filepath}")
    
        # Create and save the visualization
        create_stacked_barplot(
            output_data, 
            args.dimension,
            args.rule_type,
            args.output_filepath.split('.json')[0]+'-QAProp',
            figsize=figsize,
        )
    else:
        output_data = None

    '''
    create_l1_distance_boxplot(
        output_data, 
        args.dimension, 
        args.output_filepath.split('.json')[0]+'-L1Dist',
        figsize=figsize,
    )

    '''
    if args.dimension == 0:
        create_aggregated_l1_boxplot_with_mean(
            output_data, 
            args.rule_type,
            args.output_filepath.split('.json')[0]+'-AggL1Dist.' + args.format,
            #figsize=(20,3), 
            figsize=figsize,
            json_config=args.json_config, #**kwargs,
            args=args,
        )

if __name__ == "__main__":
    main()

