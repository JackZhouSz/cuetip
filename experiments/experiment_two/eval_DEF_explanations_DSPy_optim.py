from typing import List, Tuple, Dict, Any
import os
import json
from copy import deepcopy
import argparse
import numpy as np
import weave
import dspy
from dspy.teleprompt import (
    BootstrapFewShotWithRandomSearch,
    COPRO,
    MIPROv2,
)
from dspy import ChainOfThought
from gen_DEF_explanations import (
    ExplainDEFWithOutDEFSignature, 
    ExplainDEFWithDEFSignature, 
    ExplainDEFWithDEFPercentageSignature,
    EstimateEnum,
)

#from dspy.predict.aggregation import majority
import pydantic
import pandas as pd


class ExplainDEFWithDEFPercentageSignature_v1(dspy.Signature):
    """
    You are tasked with providing estimations of the applicability of some value and difficulty rules to a particular shot in a game of pool, using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow'
        - The shot parameters are provided, and are defined as:
            - V0: the initial velocity of the cue ball
            - theta: The angle of inclination of the cue stick
            - phi: The angle of rotation of the cue stick
            - a: The x offset of the cue stick
            - b: The y offset of the cue stick
        - The exact (x,y) coordinates of each ball and pocket on the table
        - The events that occurred in the shot, and their positions
        - The value rules and weights
        - The difficulty rules and weights

    You must return an estimate of the applicability of each of the value and difficulty rules.
    The value and difficulty rules' weights are percentages that represent the extent to which a given rule applies to the current state and shot.
    Your estimation of applicability should convert a given rule's percentage weights X as follows:
        - if 0 <= X < 12.5, then it is very low ;
        - if 12.5 <= X < 25, then it is low ;
        - if 25 <= X < 37.5, then moderately low ;
        - if 37.5 <= X < 62.5, then moderate ;
        - if 62.5 <= X < 75, then moderately high ;
        - if 75 <= X < 87.5, then high ;
        - if 87.5 <= X <= 100, then very high.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")
    #difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")

    value1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 1 to the current situation's table and shot.")
    value2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 2 to the current situation's table and shot.")
    value3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 3 to the current situation's table and shot.")
    value4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 4 to the current situation's table and shot.")
    value5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 5 to the current situation's table and shot.")
    value6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 6 to the current situation's table and shot.")
    value7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 7 to the current situation's table and shot.")
    value8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 8 to the current situation's table and shot.")
    value9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 9 to the current situation's table and shot.")
    value10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 10 to the current situation's table and shot.")
    value11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 11 to the current situation's table and shot.")
    value12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 12 to the current situation's table and shot.")
    value13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 13 to the current situation's table and shot.")

    difficulty1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 1 to the current situation's table and shot.")
    difficulty2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 2 to the current situation's table and shot.")
    difficulty3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 3 to the current situation's table and shot.")
    difficulty4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 4 to the current situation's table and shot.")
    difficulty5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 5 to the current situation's table and shot.")
    difficulty6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 6 to the current situation's table and shot.")
    difficulty7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 7 to the current situation's table and shot.")
    difficulty8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 8 to the current situation's table and shot.")
    difficulty9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 9 to the current situation's table and shot.")
    difficulty10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 10 to the current situation's table and shot.")
    difficulty11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 11 to the current situation's table and shot.")
    difficulty12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 12 to the current situation's table and shot.")
    difficulty13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 13 to the current situation's table and shot.")
    difficulty14_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 14 to the current situation's table and shot.")
    difficulty15_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 15 to the current situation's table and shot.")
    difficulty16_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 16 to the current situation's table and shot.")


class ExplainDEFWithDEFPercentageSignature_v2(dspy.Signature):
    """
    You are tasked with providing estimations of the applicability of some value and difficulty rules to a particular shot in a game of pool, using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow'
        - The shot parameters are provided, and are defined as:
            - V0: the initial velocity of the cue ball
            - theta: The angle of inclination of the cue stick
            - phi: The angle of rotation of the cue stick
            - a: The x offset of the cue stick
            - b: The y offset of the cue stick
        - The exact (x,y) coordinates of each ball and pocket on the table
        - The events that occurred in the shot, and their positions
        - The value rules and weights
        - The difficulty rules and weights

    You must return an estimate of the applicability of each of the value and difficulty rules.
    The value and difficulty rules' weights are percentages that represent the extent to which a give rule applies.
    Your estimation of applicability should convert a given rule's percentage weights X as follows:
        - if 0 <= X < 12.5, then it is very low ;
        - if 12.5 <= X < 25, then it is low ;
        - if 25 <= X < 37.5, then moderately low ;
        - if 37.5 <= X < 62.5, then moderate ;
        - if 62.5 <= X < 75, then moderately high ;
        - if 75 <= X < 87.5, then high ;
        - if 87.5 <= X <= 100, then very high.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")
    #difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")

    value1_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 1.",
        prefix="Reasoning Value Rule 1: Let's think step by step in order to",
    )
    value1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 1 to the current situation's table and shot.")
    value2_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 2.",
        prefix="Reasoning Value Rule 2: Let's think step by step in order to",
    )
    value2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 2 to the current situation's table and shot.")
    value3_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 3.",
        prefix="Reasoning Value Rule 3: Let's think step by step in order to",
    )
    value3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 3 to the current situation's table and shot.")
    value4_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 4.",
        prefix="Reasoning Value Rule 4: Let's think step by step in order to",
    )
    value4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 4 to the current situation's table and shot.")
    value5_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 5.",
        prefix="Reasoning Value Rule 5: Let's think step by step in order to",
    )
    value5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 5 to the current situation's table and shot.")
    value6_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 6.",
        prefix="Reasoning Value Rule 6: Let's think step by step in order to",
    )
    value6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 6 to the current situation's table and shot.")
    value7_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 7.",
        prefix="Reasoning Value Rule 7: Let's think step by step in order to",
    )
    value7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 7 to the current situation's table and shot.")
    value8_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 8.",
        prefix="Reasoning Value Rule 8: Let's think step by step in order to",
    )
    value8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 8 to the current situation's table and shot.")
    value9_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 9.",
        prefix="Reasoning Value Rule 9: Let's think step by step in order to",
    )
    value9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 9 to the current situation's table and shot.")
    value10_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 10.",
        prefix="Reasoning Value Rule 10: Let's think step by step in order to",
    )
    value10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 10 to the current situation's table and shot.")
    value11_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 11.",
        prefix="Reasoning Value Rule 11: Let's think step by step in order to",
    )
    value11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 11 to the current situation's table and shot.")
    value12_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 12.",
        prefix="Reasoning Value Rule 12: Let's think step by step in order to",
    )
    value12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 12 to the current situation's table and shot.")
    value13_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 13.",
        prefix="Reasoning Value Rule 13: Let's think step by step in order to",
    )
    value13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 13 to the current situation's table and shot.")

    difficulty1_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 1.",
        prefix="Reasoning Difficulty Rule 1: Let's think step by step in order to",
    )
    difficulty1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 1 to the current situation's table and shot.")
    difficulty2_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 2.",
        prefix="Reasoning Difficulty Rule 2: Let's think step by step in order to",
    )
    difficulty2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 2 to the current situation's table and shot.")
    difficulty3_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 3.",
        prefix="Reasoning Difficulty Rule 3: Let's think step by step in order to",
    )
    difficulty3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 3 to the current situation's table and shot.")
    difficulty4_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 4.",
        prefix="Reasoning Difficulty Rule 4: Let's think step by step in order to",
    )
    difficulty4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 4 to the current situation's table and shot.")
    difficulty5_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 5.",
        prefix="Reasoning Difficulty Rule 5: Let's think step by step in order to",
    )
    difficulty5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 5 to the current situation's table and shot.")
    difficulty6_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 6.",
        prefix="Reasoning Difficulty Rule 6: Let's think step by step in order to",
    )
    difficulty6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 6 to the current situation's table and shot.")
    difficulty7_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 7.",
        prefix="Reasoning Difficulty Rule 7: Let's think step by step in order to",
    )
    difficulty7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 7 to the current situation's table and shot.")
    difficulty8_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 8.",
        prefix="Reasoning Difficulty Rule 8: Let's think step by step in order to",
    )
    difficulty8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 8 to the current situation's table and shot.")
    difficulty9_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 9.",
        prefix="Reasoning Difficulty Rule 9: Let's think step by step in order to",
    )
    difficulty9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 9 to the current situation's table and shot.")
    difficulty10_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 10.",
        prefix="Reasoning Difficulty Rule 10: Let's think step by step in order to",
    )
    difficulty10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 10 to the current situation's table and shot.")
    difficulty11_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 11.",
        prefix="Reasoning Difficulty Rule 11: Let's think step by step in order to",
    )
    difficulty11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 11 to the current situation's table and shot.")
    difficulty12_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 12.",
        prefix="Reasoning Difficulty Rule 12: Let's think step by step in order to",
    )
    difficulty12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 12 to the current situation's table and shot.")
    difficulty13_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 13.",
        prefix="Reasoning Difficulty Rule 13: Let's think step by step in order to",
    )
    difficulty13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 13 to the current situation's table and shot.")
    difficulty14_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 14.",
        prefix="Reasoning Difficulty Rule 14: Let's think step by step in order to",
    )
    difficulty14_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 14 to the current situation's table and shot.")
    difficulty15_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 15.",
        prefix="Reasoning Difficulty Rule 15: Let's think step by step in order to",
    )
    difficulty15_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 15 to the current situation's table and shot.")
    difficulty16_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 16.",
        prefix="Reasoning Difficulty Rule 16: Let's think step by step in order to",
    )
    difficulty16_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 16 to the current situation's table and shot.")

class ExplainDEFWithDEFPercentageSignature_v3(dspy.Signature):
    """
    You are tasked with providing estimations of the applicability of some value and difficulty rules to a particular shot in a game of pool, using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow'
        - The shot parameters are provided, and are defined as:
            - V0: the initial velocity of the cue ball
            - theta: The angle of inclination of the cue stick
            - phi: The angle of rotation of the cue stick
            - a: The x offset of the cue stick
            - b: The y offset of the cue stick
        - The exact (x,y) coordinates of each ball and pocket on the table
        - The events that occurred in the shot, and their positions
        - The value rules and weights
        - The difficulty rules and weights

    You must return an estimate of the applicability of each of the value and difficulty rules.
    The value and difficulty rules' weights are percentages that represent the extent to which a given rule applies to the current state and shot.
    To convert a given rule's percentage weight X into a valid estimation of the applicability of the given rule to the current state and shot, proceed as follows:
        - if 0 <= X < 12.5, then the given rule's applicability is very low ;
        - if 12.5 <= X < 25, then the given rule's applicability is low ;
        - if 25 <= X < 37.5, then the given rule's applicability is moderately low ;
        - if 37.5 <= X < 62.5, then the given rule's applicability is moderate ;
        - if 62.5 <= X < 75, then the given rule's applicability is moderately high ;
        - if 75 <= X < 87.5, then the given rule's applicability is high ;
        - if 87.5 <= X <= 100, then the given rule's applicability is very high.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")
    #difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")

    value1_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 1.",
        prefix="Reasoning Value Rule 1: Let's think step by step in order to",
    )
    value1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 1 to the current situation's table and shot.")
    value2_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 2.",
        prefix="Reasoning Value Rule 2: Let's think step by step in order to",
    )
    value2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 2 to the current situation's table and shot.")
    value3_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 3.",
        prefix="Reasoning Value Rule 3: Let's think step by step in order to",
    )
    value3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 3 to the current situation's table and shot.")
    value4_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 4.",
        prefix="Reasoning Value Rule 4: Let's think step by step in order to",
    )
    value4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 4 to the current situation's table and shot.")
    value5_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 5.",
        prefix="Reasoning Value Rule 5: Let's think step by step in order to",
    )
    value5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 5 to the current situation's table and shot.")
    value6_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 6.",
        prefix="Reasoning Value Rule 6: Let's think step by step in order to",
    )
    value6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 6 to the current situation's table and shot.")
    value7_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 7.",
        prefix="Reasoning Value Rule 7: Let's think step by step in order to",
    )
    value7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 7 to the current situation's table and shot.")
    value8_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 8.",
        prefix="Reasoning Value Rule 8: Let's think step by step in order to",
    )
    value8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 8 to the current situation's table and shot.")
    value9_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 9.",
        prefix="Reasoning Value Rule 9: Let's think step by step in order to",
    )
    value9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 9 to the current situation's table and shot.")
    value10_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 10.",
        prefix="Reasoning Value Rule 10: Let's think step by step in order to",
    )
    value10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 10 to the current situation's table and shot.")
    value11_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 11.",
        prefix="Reasoning Value Rule 11: Let's think step by step in order to",
    )
    value11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 11 to the current situation's table and shot.")
    value12_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 12.",
        prefix="Reasoning Value Rule 12: Let's think step by step in order to",
    )
    value12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 12 to the current situation's table and shot.")
    value13_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of value rule 13.",
        prefix="Reasoning Value Rule 13: Let's think step by step in order to",
    )
    value13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 13 to the current situation's table and shot.")

    difficulty1_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 1.",
        prefix="Reasoning Difficulty Rule 1: Let's think step by step in order to",
    )
    difficulty1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 1 to the current situation's table and shot.")
    difficulty2_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 2.",
        prefix="Reasoning Difficulty Rule 2: Let's think step by step in order to",
    )
    difficulty2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 2 to the current situation's table and shot.")
    difficulty3_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 3.",
        prefix="Reasoning Difficulty Rule 3: Let's think step by step in order to",
    )
    difficulty3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 3 to the current situation's table and shot.")
    difficulty4_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 4.",
        prefix="Reasoning Difficulty Rule 4: Let's think step by step in order to",
    )
    difficulty4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 4 to the current situation's table and shot.")
    difficulty5_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 5.",
        prefix="Reasoning Difficulty Rule 5: Let's think step by step in order to",
    )
    difficulty5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 5 to the current situation's table and shot.")
    difficulty6_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 6.",
        prefix="Reasoning Difficulty Rule 6: Let's think step by step in order to",
    )
    difficulty6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 6 to the current situation's table and shot.")
    difficulty7_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 7.",
        prefix="Reasoning Difficulty Rule 7: Let's think step by step in order to",
    )
    difficulty7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 7 to the current situation's table and shot.")
    difficulty8_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 8.",
        prefix="Reasoning Difficulty Rule 8: Let's think step by step in order to",
    )
    difficulty8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 8 to the current situation's table and shot.")
    difficulty9_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 9.",
        prefix="Reasoning Difficulty Rule 9: Let's think step by step in order to",
    )
    difficulty9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 9 to the current situation's table and shot.")
    difficulty10_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 10.",
        prefix="Reasoning Difficulty Rule 10: Let's think step by step in order to",
    )
    difficulty10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 10 to the current situation's table and shot.")
    difficulty11_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 11.",
        prefix="Reasoning Difficulty Rule 11: Let's think step by step in order to",
    )
    difficulty11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 11 to the current situation's table and shot.")
    difficulty12_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 12.",
        prefix="Reasoning Difficulty Rule 12: Let's think step by step in order to",
    )
    difficulty12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 12 to the current situation's table and shot.")
    difficulty13_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 13.",
        prefix="Reasoning Difficulty Rule 13: Let's think step by step in order to",
    )
    difficulty13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 13 to the current situation's table and shot.")
    difficulty14_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 14.",
        prefix="Reasoning Difficulty Rule 14: Let's think step by step in order to",
    )
    difficulty14_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 14 to the current situation's table and shot.")
    difficulty15_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 15.",
        prefix="Reasoning Difficulty Rule 15: Let's think step by step in order to",
    )
    difficulty15_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 15 to the current situation's table and shot.")
    difficulty16_reasoning : str = dspy.OutputField(
        desc="reasoning for the estimation of the applicability of difficulty rule 16.",
        prefix="Reasoning Difficulty Rule 16: Let's think step by step in order to",
    )
    difficulty16_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 16 to the current situation's table and shot.")


class ExplainDEFWithDEFPercentageSignature_v4(dspy.Signature):
    """
    You are tasked with providing estimations of the applicability of some value and difficulty rules to a particular shot in a game of pool, using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow'
        - The shot parameters are provided, and are defined as:
            - V0: the initial velocity of the cue ball
            - theta: The angle of inclination of the cue stick
            - phi: The angle of rotation of the cue stick
            - a: The x offset of the cue stick
            - b: The y offset of the cue stick
        - The exact (x,y) coordinates of each ball and pocket on the table
        - The events that occurred in the shot, and their positions
        - The value rules and weights
        - The difficulty rules and weights

    You must return an estimate of the applicability of each of the value and difficulty rules.
    The value and difficulty rules' weights are percentages that represent the extent to which a given rule applies to the current state and shot.
    To convert a given rule's percentage weight X into a valid estimation of the applicability of the given rule to the current state and shot, proceed as follows:
        - if 0 <= X < 12.5, then the given rule's applicability is very low ;
        - if 12.5 <= X < 25, then the given rule's applicability is low ;
        - if 25 <= X < 37.5, then the given rule's applicability is moderately low ;
        - if 37.5 <= X < 62.5, then the given rule's applicability is moderate ;
        - if 62.5 <= X < 75, then the given rule's applicability is moderately high ;
        - if 75 <= X < 87.5, then the given rule's applicability is high ;
        - if 87.5 <= X <= 100, then the given rule's applicability is very high.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")
    #difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")

    value1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 1 to the current situation's table and shot.")
    value2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 2 to the current situation's table and shot.")
    value3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 3 to the current situation's table and shot.")
    value4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 4 to the current situation's table and shot.")
    value5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 5 to the current situation's table and shot.")
    value6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 6 to the current situation's table and shot.")
    value7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 7 to the current situation's table and shot.")
    value8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 8 to the current situation's table and shot.")
    value9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 9 to the current situation's table and shot.")
    value10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 10 to the current situation's table and shot.")
    value11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 11 to the current situation's table and shot.")
    value12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 12 to the current situation's table and shot.")
    value13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of value rule 13 to the current situation's table and shot.")

    difficulty1_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 1 to the current situation's table and shot.")
    difficulty2_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 2 to the current situation's table and shot.")
    difficulty3_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 3 to the current situation's table and shot.")
    difficulty4_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 4 to the current situation's table and shot.")
    difficulty5_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 5 to the current situation's table and shot.")
    difficulty6_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 6 to the current situation's table and shot.")
    difficulty7_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 7 to the current situation's table and shot.")
    difficulty8_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 8 to the current situation's table and shot.")
    difficulty9_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 9 to the current situation's table and shot.")
    difficulty10_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 10 to the current situation's table and shot.")
    difficulty11_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 11 to the current situation's table and shot.")
    difficulty12_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 12 to the current situation's table and shot.")
    difficulty13_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 13 to the current situation's table and shot.")
    difficulty14_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 14 to the current situation's table and shot.")
    difficulty15_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 15 to the current situation's table and shot.")
    difficulty16_est : EstimateEnum = dspy.OutputField(desc="an estimation of the applicability of difficulty rule 16 to the current situation's table and shot.")

def majority(lv:List):
    # Determine the most-voted enum
    lv_counts = pd.Series(lv).value_counts()
    most_voted = lv_counts.idxmax()
    return most_voted

class SCCoTModule(dspy.Module):
    def __init__(self, signature, n_samples=10, max_error=3):
        dspy.Module.__init__(self)
        self.signature = signature
        self.n_samples = n_samples
        self.max_error = max_error
        self.cot_prog = dspy.ChainOfThought(signature, n=self.n_samples)
        self.aggregation = majority

    def forward(self, **kwargs)->List[object]:
        predictions = {}
        preds = None
        error_count = 0
        while preds is None:
            try:
                preds = self.cot_prog(**{k:kwargs[k] for k in self.signature.input_fields.keys()})
            except Exception as e:
                error_count += 1
                print(f"Exception {error_count}: {e}")
                kwargs['seed'] += 1
            if error_count >= self.max_error:
                break
        if preds is not None:
            for pred in preds.completions:
                for k,v in pred.items():
                    if k not in predictions: predictions[k] = []
                    predictions[k].append(v)
        else:
            #predictions = {k:[EstimateEnum.UNKNOWN] for k in self.signature.output_fields.keys()}
            predictions = {k:[EstimateEnum.unknown] for k in self.signature.output_fields.keys()}
        output_dict = {}
        for k,lv in predictions.items():
            output_dict[k] = self.aggregation(lv)
        return output_dict

# Helper: Map EstimationEnum to Likert scale
LIKERT_SCALE = {
    EstimateEnum.very_low: 0,
    EstimateEnum.low: 1,
    EstimateEnum.moderately_low: 2,
    EstimateEnum.moderate: 3,
    EstimateEnum.moderately_high: 4,
    EstimateEnum.high: 5,
    EstimateEnum.very_high: 6,
    #EstimateEnum.UNKNOWN: EstimateEnum.UNKNOWN,
    EstimateEnum.unknown: EstimateEnum.unknown,
}


def estimation_to_likert(estimation: EstimateEnum) -> int:
    """
    Convert an EstimationEnum to a Likert scale value.

    Args:
        estimation (EstimateEnum): Enum value.

    Returns:
        int: Likert scale value.
    """
    return LIKERT_SCALE[estimation]


def load_dataset_from_weave(
    task_name: str, 
    seed:int,
    value_rules_entry:str,
    difficulty_rules_entry:str,
):
    """
    Load the training, validation, and testing datasets from Weave.

    Args:
        task_name (str): Task name for the dataset stored in Weave.
        seed (int): Seed of the experiment
        value_rules_entry (str): DSPy signature entry string for value rules and optional weights.
        difficulty_rules_entry (str): DSPy signature entry string for diffuclty rules and optional weights.
    Returns:
        Tuple of training, validation, and testing datasets of DSPy Examples.
    """
    try:
        train_dataset = weave.ref(f"{task_name}_train").get()
        val_dataset = weave.ref(f"{task_name}_val").get()
        test_dataset = weave.ref(f"{task_name}_test").get()
        print(f"Loaded datasets:")
        print(f"- {len(train_dataset.rows)} training examples;")
        print(f"- {len(val_dataset.rows)} validation examples;")
        print(f"- {len(test_dataset.rows)} testing examples;")
    except Exception as e:
        raise RuntimeError(f"Failed to retrieve datasets from Weave: {e}")
    
    # Preprocessing:
    train_dataset = [
        dspy.Example(
            {**ex, 'seed':seed}
        ).with_inputs(
            'seed',
            'shot_params',
            'board_coordinates',
            'events',
            value_rules_entry,
            difficulty_rules_entry,
        )
        for ex in train_dataset.rows
    ]
    val_dataset = [
        dspy.Example(
            {**ex, 'seed':seed}
        ).with_inputs(
            'seed',
            'shot_params',
            'board_coordinates',
            'events',
            value_rules_entry,
            difficulty_rules_entry,
        )
        for ex in val_dataset.rows
    ]
    test_dataset = [
        dspy.Example(
            {**ex, 'seed':seed}
        ).with_inputs(
            'seed',
            'shot_params',
            'board_coordinates',
            'events',
            value_rules_entry,
            difficulty_rules_entry,
        )
        for ex in test_dataset.rows
    ]
    return train_dataset, val_dataset, test_dataset


def l1_dist_fn(target,pred):
    if pred != EstimateEnum.unknown:
        dist = abs(target-pred)
    else:
        dist = 6
    return dist


@weave.op()
def compute_l1_distance(example, prediction, trace=None):
    """
    Compute the L1 distance between model predictions and target answers.

    Args:
        example: Ground truth example with targets.
        prediction: Model prediction with estimated values.

    Returns:
        float: Mean L1 distance across all rules.
    """
    l1_distances = []

    # Compute L1 distance for value rules
    for idx in range(1, 14):  # Assuming 13 value rules
        target = estimation_to_likert(EstimateEnum[example['target_answer']['value'][f"value{idx}_est"]])
        pred = estimation_to_likert(prediction[f"value{idx}_est"])
        #if pred != EstimateEnum.UNKNOWN:
        if pred != EstimateEnum.unknown:
            dist = abs(target-pred)
        else:
            dist = 6
        l1_distances.append(dist)

    # Compute L1 distance for difficulty rules
    for idx in range(1, 17):  # Assuming 16 difficulty rules
        target = estimation_to_likert(EstimateEnum[example['target_answer']['difficulty'][f"difficulty{idx}_est"]])
        pred = estimation_to_likert(prediction[f"difficulty{idx}_est"])
        #if pred != EstimateEnum.UNKNOWN:
        if pred != EstimateEnum.unknown:
            dist = abs(target-pred)
        else:
            dist = 6
        l1_distances.append(dist)

    #return -np.mean(l1_distances)
    #return 1.0/(1e-1+np.mean(l1_distances))
    return 1.0 - np.mean(l1_distances)/6

@weave.op()
def compute_accuracy(example, prediction, trace=None):
    """
    Compute the accuracy between model predictions and target answers.

    Args:
        example: Ground truth example with targets.
        prediction: Model prediction with estimated values.

    Returns:
        float: accuracy across all rules.
    """
    l1_distances = []

    # Compute L1 distance for value rules
    for idx in range(1, 14):  # Assuming 13 value rules
        target = estimation_to_likert(EstimateEnum[example['target_answer']['value'][f"value{idx}_est"]])
        pred = estimation_to_likert(prediction[f"value{idx}_est"])
        #if pred != EstimateEnum.UNKNOWN:
        if pred != EstimateEnum.unknown:
            dist = abs(target-pred)
        else:
            dist = 6
        l1_distances.append(dist)

    # Compute L1 distance for difficulty rules
    for idx in range(1, 17):  # Assuming 16 difficulty rules
        target = estimation_to_likert(EstimateEnum[example['target_answer']['difficulty'][f"difficulty{idx}_est"]])
        pred = estimation_to_likert(prediction[f"difficulty{idx}_est"])
        #if pred != EstimateEnum.UNKNOWN:
        if pred != EstimateEnum.unknown:
            dist = abs(target-pred)
        else:
            dist = 6
        l1_distances.append(dist)
    l1_distances = np.array(l1_distances)
    return np.mean(l1_distances==0)


def load_module(
    module_path: str, 
    signature:dspy.Signature, 
    args:Dict,
):
    """
    Load a previously saved module locally.

    Args:
        module_path (str): path to the module.
        signature (dspy.Signature): DSPy signature to use with the module.
        args : Argparse arguments

    Returns:
        Loaded DSPy module.
    """
    module = SCCoTModule(
        signature=signature,
        n_samples=args.n_samples,
    )
    try:
        module.load(module_path)
        print(f"Successfully loaded module '{module_path}'.")
        return module
    except Exception as e:
        raise RuntimeError(f"Failed to load module '{module_path}': {e}")


@weave.op
def optimise_with_teleprompter(
    trainset, 
    devset, 
    signature, 
    teleprompter_type,
    n_threads,
    args,
):
    """
    Optimise a DSPy ChainOfThought module using a teleprompter.
    Follow doc: https://dspy.ai/learn/optimization/optimizers/
    Args:
        trainset: Optimising examples.
        devset: Validation examples.
        signature: DSPy signature for the ChainOfThought module.
        teleprompter_type (str): Teleprompter type ('bootstrap', 'mipro').
        n_threads (int): number of threads to use in optimisation.

    Returns:
        Optimized ChainOfThought module.
    """
    # Initialize the ChainOfThought module
    #dspy_module = ChainOfThought(signature)
    dspy_module = SCCoTModule(
        signature=signature,
        n_samples=args.n_samples,
    )
    
    metric_fn = compute_l1_distance
    #metric_fn = compute_accuracy
    compile_kwargs = dict(
        trainset=trainset[:50],#TODO, 
        #requires_permission_to_run=False,
    )

    # Select the teleprompter
    if teleprompter_type == "bootstrap":
        # Set up the optimizer: 
        # we want to "bootstrap" (i.e., self-generate) 4-shot examples of your program's steps.
        # The optimizer will repeat this 10 times (plus some initial attempts) 
        # before selecting its best attempt on the devset.
        config = dict(
            max_bootstrapped_demos=args.nbr_demos, 
            max_labeled_demos=args.nbr_demos, 
            num_candidate_programs=10, 
            num_threads=n_threads,
        )
        teleprompter = BootstrapFewShotWithRandomSearch(
            metric=metric_fn,
            **config,
        )
    elif "mipro" in teleprompter_type :
        config = dict(
            auto="light", #medium #heavy 
            num_threads=n_threads,
            max_bootstrapped_demos=args.nbr_demos, #0, # ZERO FEW-SHOT EXAMPLES
            max_labeled_demos=args.nbr_demos, #0, # ZERO FEW-SHOT EXAMPLES
        )
        teleprompter = MIPROv2(
            metric=metric_fn,
            verbose=True,
            **config,
        )
        compile_kwargs['requires_permission_to_run'] = False
    elif "copro" in teleprompter_type:
        config = dict(
            num_threads=n_threads,
            breadth=10,
            depth=3,
            init_temperature=args.init_temperature,
            track_stats=False,
        )
        teleprompter = COPRO(
            metric=metric_fn,
            verbose=True,
            **config,
        )
        eval_kwargs = dict(
            num_threads=n_threads, 
            display_progress=True, 
            display_table=2,
        )
        compile_kwargs['eval_kwargs'] = eval_kwargs
    else:
        raise ValueError(f"Unknown teleprompter type: {teleprompter_type}")

    #example = trainset[0]
    #prediction = dspy_module(**example)
    #eval_results = compute_l1_distance(example,prediction)
    #eval_results = compute_accuracy(example,prediction)
    
    # Compile the optimized module
    print(f"Optimising the ChainOfThought module with {teleprompter_type} teleprompter...")
    compiled_module = teleprompter.compile(
        deepcopy(dspy_module), 
        **compile_kwargs,
    )
    print("Optimising completed.")

    return compiled_module


@weave.op
def evaluate_with_l1(module, evalset, args):
    """
    Evaluate the optimized module using mean L1 distance.

    Args:
        module: Optimized ChainOfThought module.
        evalset: Evaluation dataset examples.

    Returns:
        float: Mean L1 distance across the validation set.
    """
    print("Evaluating the optimized module with L1 distance...")
    '''
    distances = [compute_l1_distance(example, module(**example)) for example in evalset]
    mean_distance = np.mean(distances)
    print(f"Mean L1 Distance: {mean_distance}")
    return mean_distance
    '''
    '''
    evaluation = weave.Evaluation(
        name=args.module_path.split('/')[-1],
        dataset=[{**e} for e in evalset],
        scorers=[compute_l1_distance, compute_accuracy],
    )
    evaluation.evaluate(module.forward)
    '''
    evaluate = dspy.evaluate.Evaluate(
        devset=evalset, 
        metric=compute_l1_distance,
        num_threads=args.n_threads, 
        display_progress=True, 
        display_table=10,
        return_outputs=True,
    )

    score, results = evaluate(module)
    return results


def save_module_with_weave(module, project_name: str, module_name: str):
    """
    Save the optimized module to Weave.

    Args:
        module: Optimized DSPy module.
        project_name (str): Weave project name.
        module_name (str): Name for the module in Weave.
    """
    weave.publish(
        module,
        name=module_name,
    )
    print(f"Module saved to Weave under project '{project_name}' with name '{module_name}'.")


def save_and_format_results(
    results:List[List[Any]],
    args:Dict[str,Any],
):
    ''' 
    Format the list of results from a dspy.evaluate.Evaluate method with return_outputs=True,
    and then saves it in a result folder.

    :param results: List[List[ ]] of following elements:
        0. DSPy Example
        1. Model Prediction as Dict
        2. output of the metric fn with the previous two arguments.

    :returns X_data: with X in ['value','difficulty'] for each type of rules with following arch:
        Dict[
            rule_idx : Dict[
                args.module_path's name/method: {
                    'l1_distances': List[float]
                }
            ]
        ]
    '''
    #TODO: update catefory2legend dict with copro+heuristics/baseline etc...
    # retrieve method/category name from module_path:
    category = args.module_path.split('/')[-1]

    # Format as List of nbr_rules Dict with entries:
    # -with_functions 
    # -without_functions , i.e. the method used
    # and the values are Dict with at least entry:
    # l1_distances -> List[float]
    # Separate value from difficulty rules:
    value_data = {}
    difficulty_data = {}
    for ridx, (dspy_ex, pred, eval_out) in enumerate(results):
        for rule_idx in range(17):
            if f"value{rule_idx}_est" in pred:
                kdata = value_data
                if rule_idx not in kdata:
                    kdata[rule_idx] = {}
                if category not in kdata[rule_idx]:
                    kdata[rule_idx][category] = {}
                if 'l1_distances' not in kdata[rule_idx][category]:
                    kdata[rule_idx][category]['l1_distances'] = []
                dist =  l1_dist_fn(
                    target=estimation_to_likert(
                        EstimateEnum[dspy_ex['target_answer']['value'][f"value{rule_idx}_est"]],
                    ),
                    pred=estimation_to_likert(pred[f"value{rule_idx}_est"]),
                )
                kdata[rule_idx][category]['l1_distances'].append(dist)
        
            if f"difficulty{rule_idx}_est" in pred:
                kdata = difficulty_data
                if rule_idx not in kdata:
                    kdata[rule_idx] = {}
                if category not in kdata[rule_idx]:
                    kdata[rule_idx][category] = {}
                if 'l1_distances' not in kdata[rule_idx][category]:
                    kdata[rule_idx][category]['l1_distances'] = []
                dist =  l1_dist_fn(
                    target=estimation_to_likert(
                        EstimateEnum[dspy_ex['target_answer']['difficulty'][f"difficulty{rule_idx}_est"]],
                    ),
                    pred=estimation_to_likert(pred[f"difficulty{rule_idx}_est"]),
                )
                kdata[rule_idx][category]['l1_distances'].append(dist)
    
    #value_data = list(value_data.values())
    #difficulty_data = list(difficulty_data.values())

    base_path = args.module_path.split('/')
    name_path = base_path[-1]
    base_path = "/".join(base_path[:-1])

    # Save values:
    save_path = base_path+"/"+f"results-values-"+name_path
    print(f"Saving value results at : {save_path}")
    with open(save_path, 'w') as f:
        json.dump(value_data, f, indent=2)

    # Save difficulty:
    save_path = base_path+"/"+f"results-difficulties-"+name_path
    print(f"Saving difficulty results at : {save_path}")
    with open(save_path, 'w') as f:
        json.dump(difficulty_data, f, indent=2)

    return value_data, difficulty_data

def str2bool(inp):
    if not isinstance(inp, bool):
        assert isinstance(inp, str)
        inp = 'true' in inp.lower()
    return inp


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Optimize and evaluate a DSPy ChainOfThought module.")
    parser.add_argument("--seed", type=int, default=10, help="Seed of the experiment.")
    parser.add_argument("--task_name", type=str, required=True, help="The task name used to store datasets in Weave.")
    parser.add_argument("--project_name", type=str, required=True, help="Weave project name.")
    parser.add_argument("--module_path", type=str, required=True, help="Path for the module in local.")
    parser.add_argument("--model_id", type=str, default="", help="Choose model_id for Hugging Face models")
    parser.add_argument("--model_type", type=str, default="chat", help="Choose model type template among [chat,text]")
    parser.add_argument("--port", type=int, default=7501, help="Port on which to connect to query the local Hugging Face models")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature to use LLMs with.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max number of tokens to use when generating tokens with LLMs.")
    parser.add_argument("--n_samples", type=int, default=3, help="Majority voting number of samples.")
    parser.add_argument("--nbr_demos", type=int, default=0, help="Number of Few-Shot Examples to optimise for, either with BootstrapFewShotWithRandomSearch or with MIPROv2.")
    parser.add_argument("--reset", type=str2bool, default=False, help="Optimize the module anew if set.")
    parser.add_argument("--with_def_signature", type=str2bool, default=False, help="Whether to optimise and/or evaluate for ExplainDEFWithDEFSignature.")
    parser.add_argument(
        "--teleprompter_type", 
        type=str, 
        default="none", 
        choices=["none", "bootstrap", "copro", "mipro"], 
        help="Type of teleprompter to use: BoostrapFewShotWithRandomSearch will search for programs by optimising few-shot examples / MIPROv2 will optimise the DSPy signature and optionally the few-shot examples.",
    )
    parser.add_argument("--n_threads", type=int, default=4, help="Number of concurrent threads")
    parser.add_argument("--init_temperature", type=float, default=0.9, help="Init temperature hyperparameter used in COPRO optimiser. DSPy default was 1.4 which result in unparse-able answers from LM, it has therefore been reduced to 0.9, just above the usual LM default=0.7.")
    args = parser.parse_args()

    # Initialize Weave
    weave.init(project_name=args.project_name)

    # HuggingFace API client:
    from huggingface_hub import login
    assert "HUGGINGFACE_TOKEN" in os.environ, "Please set the Hugging Face token as an environment variable"
    HF_TOKEN = os.environ["HUGGINGFACE_TOKEN"]
    login(HF_TOKEN)

    api_key = os.getenv("HUGGINGFACE_TOKEN")
    config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    
    llm = dspy.LM(
        model = 'openai/'+args.model_id,
        api_base=f'http://localhost:{args.port}/v1',
        api_key='local',
        model_type=args.model_type, #'chat',
        **config,
    ) 
    dspy.settings.configure(lm=llm)

    value_rules_entry = "value_function_rules"
    difficulty_rules_entry = "difficulty_function_rules"
    if args.with_def_signature:
        value_rules_entry = "value_function_rules_vals"
        difficulty_rules_entry = "difficulty_function_rules_vals"

    testset = None
    if args.reset and args.teleprompter_type != "none":
        print("Reset flag detected. Optimising a new module...")
        # Load dataset
        trainset, devset, testset = load_dataset_from_weave(
            args.task_name, 
            seed=args.seed,
            value_rules_entry=value_rules_entry,
            difficulty_rules_entry=difficulty_rules_entry,
        )

        # Optimise the module using the selected teleprompter
        optimized_module = optimise_with_teleprompter(
            trainset=trainset, 
            devset=devset, 
            #signature=ExplainDEFWithDEFSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            #signature=ExplainDEFWithDEFPercentageSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            signature=ExplainDEFWithDEFPercentageSignature_v4 if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            teleprompter_type=args.teleprompter_type,
            n_threads=args.n_threads,
            args=args,
        )

        # Save the optimized module
        '''
        save_module(
            optimized_module, 
            project_name=args.project_name, 
            module_path=args.module_path,
        )
        '''
        optimized_module.save(args.module_path)
    elif args.teleprompter_type =="none":
        optimized_module = SCCoTModule(
            #signature=ExplainDEFWithDEFSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            #signature=ExplainDEFWithDEFPercentageSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            signature=ExplainDEFWithDEFPercentageSignature_v4 if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            n_samples=args.n_samples,
        )
    else:
        print("Loading previously saved module from Weave...")
        optimized_module = load_module(
            module_path=args.module_path,
            #signature=ExplainDEFWithDEFSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            #signature=ExplainDEFWithDEFPercentageSignature if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            signature=ExplainDEFWithDEFPercentageSignature_v4 if args.with_def_signature else ExplainDEFWithOutDEFSignature, 
            args=args,
        )

    # Evaluate the optimized module
    if testset is None:
        trainset, devset, testset = load_dataset_from_weave(
            args.task_name, 
            seed=args.seed,
            value_rules_entry=value_rules_entry,
            difficulty_rules_entry=difficulty_rules_entry,
        )
    
    results = evaluate_with_l1(
        module=optimized_module, 
        evalset=testset,
        args=args,
    )
    
    value_data, difficulty_data = save_and_format_results(
        results=results,
        args=args,
    )
    # Finalize Weave session
    weave.finish()


if __name__ == "__main__":
    main()

