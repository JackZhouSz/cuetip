import argparse, sys, os, logging, time, dspy, traceback, json, yaml, re
import numpy as np
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import weave
from typing import List
from pydantic import BaseModel, Field
from pathlib import Path
from enum import Enum
from pydantic import BaseModel, Field

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from poolagent.path import DATA_DIR, ROOT_DIR, RULES_DIR
from poolagent.pool import Pool
from poolagent.agents import FunctionChooser
from poolagent.utils import State, Event
from poolagent.experiment_manager import LLM

# Constants
GPU_SIZE = 40
MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2
STOP_LIST = []

class EstimateEnum(Enum):
    very_low=           "very low"
    low=                "low"
    moderately_low=     "moderately low"
    moderate=           "moderate"
    moderately_high=    "moderately high"
    high=               "high"
    very_high=          "very high"

class ExplainDEFWithOutDEFSignature(dspy.Signature):
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
        - The value rules
        - The difficulty rules

    You must return an estimate of the applicability of each of the value and difficulty rules.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    value_function_rules = dspy.InputField(desc="The rules that contribute to the value of the shot.")
    difficulty_function_rules = dspy.InputField(desc="The rules that contribute to the difficulty of the shot.")

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


class ExplainDEFWithDEFSignature(dspy.Signature):
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

    You must return an estimate of the applicability of each of the value and difficulty rules. This is very important: *the weights of each rule is directly correlated to the applicability of the rule*.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")

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

class ExplainDEFWithDEFPercentageSignature(dspy.Signature):
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

class ExplainDEFWithDEFPercentageSignature(dspy.Signature):
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

class ExplainDEFWithOutDEFSignature_v0(dspy.Signature):
    """You are tasked with providing explanations of how well some value and difficulty rules apply to a particular shot in a game of pool, using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow'
        - The shot parameters are provided, and are defined as:
            - V0: the initial velocity of the cue ball
            - theta: The angle of inclination of the cue stick
            - phi: The angle of rotation of the cue stick
            - a: The x offset of the cue stick
            - b: The y offset of the cue stick
        - The exact (x,y) coordinates of each ball and pocket on the table
        - The events that occurred in the shot, and their positions
        - The value rules
        - The difficulty rules

    You must return explanations for all of the value and difficulty rules.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    value_function_rules = dspy.InputField(desc="The rules that contribute to the value of the shot.")
    difficulty_function_rules = dspy.InputField(desc="The rules that contribute to the difficulty of the shot.")

    value1_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 1 applies to the current shot.")
    value2_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 2 applies to the current shot.")
    value3_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 3 applies to the current shot.")
    value4_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 4 applies to the current shot.")
    value5_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 5 applies to the current shot.")
    value6_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 6 applies to the current shot.")
    value7_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 7 applies to the current shot.")
    value8_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 8 applies to the current shot.")
    value9_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 9 applies to the current shot.")
    value10_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 10 applies to the current shot.")
    value11_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 11 applies to the current shot.")
    value12_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 12 applies to the current shot.")
    value13_expl : str = dspy.OutputField(desc="an explanation of the extent to which value rule 13 applies to the current shot.")

    difficulty1_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 1 applies to the current shot.")
    difficulty2_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 2 applies to the current shot.")
    difficulty3_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 3 applies to the current shot.")
    difficulty4_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 4 applies to the current shot.")
    difficulty5_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 5 applies to the current shot.")
    difficulty6_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 6 applies to the current shot.")
    difficulty7_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 7 applies to the current shot.")
    difficulty8_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 8 applies to the current shot.")
    difficulty9_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 9 applies to the current shot.")
    difficulty10_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 10 applies to the current shot.")
    difficulty11_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 11 applies to the current shot.")
    difficulty12_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 12 applies to the current shot.")
    difficulty13_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 13 applies to the current shot.")
    difficulty14_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 14 applies to the current shot.")
    difficulty15_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 15 applies to the current shot.")
    difficulty16_expl : str = dspy.OutputField(desc="an explanation of the extent to which difficulty rule 16 applies to the current shot.")

def explain_DEF_without_DEF_fn(
    llm, 
    state: State, 
    shot_params : dict, 
    events : List[Event], 
    end_state: State, 
    seed : int = 0,
    **kwargs,
) -> str:

    ### Load DSPy
    dspy.settings.configure(lm=llm)
    dspy_explain_DEF = dspy.TypedChainOfThought(ExplainDEFWithOutDEFSignature)

    shot_params_str = ""
    for key, value in shot_params.items():
        shot_params_str += f"{key}: {value:.2f}\n"

    ### Load Rules and collect the top 3 and bottom 3 rules
    value_function_rules = {}
    with open(f"{RULES_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    difficulty_function_rules = {}
    with open(f"{RULES_DIR}/difficulty_rules.json") as f:
        difficulty_function_rules = json.load(f)

    value_function_values = [
        f"Value Rule {kidx+1} --> [{value_function_rules[k]}]" 
        for kidx, k in enumerate(value_function_rules.keys())
    ]
    difficulty_function_values = [
        f"Difficulty Rule {kidx+1} --> [{difficulty_function_rules[k]}]" 
        for kidx, k in enumerate(difficulty_function_rules.keys())
    ]


    value_function_str = "\n============================================\n"
    value_function_str += "--- Value Rules ---\n"
    value_function_str += "\n".join(value_function_values)
    
    difficulty_function_str = "--- Difficulty Rules ---\n"
    difficulty_function_str += "\n".join(difficulty_function_values)
    difficulty_function_str += "\n============================================\n"

    ### Board coordinates
    board_coordinates_str = ""
    board_coordinates_str += f"Balls:\n"
    for k, v in state.ball_positions.items():
        if isinstance(v[0], str):
            continue
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"
    board_coordinates_str += f"Pockets:\n"
    for k, v in state.pocket_positions.items():
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"

    ### Events
    events_str = ""
    events_str += f"Events:\n"
    for e in events:
        events_str += f"'{e.encoding}' at {e.pos}\n"
    ### Estimate weights:
    response = dspy_explain_DEF(
        seed=f"{seed}",
        shot_params=shot_params_str,
        board_coordinates=board_coordinates_str,
        events=events_str,
        value_function_rules=value_function_str,
        difficulty_function_rules=difficulty_function_str
    )
    return response


def explain_DEF_with_DEF_fn(
    llm, 
    state: State, 
    shot_params : dict, 
    events : List[Event], 
    end_state: State, 
    value_rules_weights : List[float], 
    difficulty_rules_weights: List[float], 
    seed : int = 0,
    percentage_values : bool = False,
) -> str:

    """Use a LLM to explain a shot using the provided value and difficulty function values.

    Args:
        llm_config (dict): LLM configuration
        shot_params (dict): Shot parameters
        value_rules_weights (List[float]): List of value function weights
        difficulty_rules_weights (List[float]): List of difficulty function weights
    """

    ### Load DSPy
    dspy.settings.configure(lm=llm)
    if percentage_values:
        dspy_explain_DEF = dspy.TypedChainOfThought(ExplainDEFWithDEFPercentageSignature)
    else:
        dspy_explain_DEF = dspy.TypedChainOfThought(ExplainDEFWithDEFSignature)
    
    '''
    # Retry is not functional, unfortunately:
    dspy.primitives.assertions.assert_transform_module(
        dspy_explain_DEF.map_named_predictors(dspy.predict.retry.Retry), 
        dspy.primitives.assertions.backtrack_handler,
    )
    '''
    shot_params_str = ""
    for key, value in shot_params.items():
        shot_params_str += f"{key}: {value:.2f}\n"

    ### Load Rules and collect the top 3 and bottom 3 rules
    value_function_rules = {}
    with open(f"{RULES_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    difficulty_function_rules = {}
    with open(f"{RULES_DIR}/difficulty_rules.json") as f:
        difficulty_function_rules = json.load(f)

    if percentage_values:
        value_function_values = [
            f"Value Rule {int(i+1)} --> [{value_function_rules[str(i+1)]}] with weight of {int(w)}%" 
            for i, w in enumerate(value_rules_weights)
        ]
        difficulty_function_values = [
            f"Difficulty Rule {int(i+1)} --> [{difficulty_function_rules[str(i+1)]}] with weight of {int(w)}%" 
            for i, w in enumerate(difficulty_rules_weights)
        ]
    else:
        value_function_values = [
            f"Value Rule {int(i+1)} --> [{value_function_rules[str(i+1)]}] with weight {float(w):.2f}" 
            for i, w in enumerate(value_rules_weights)
        ]
        difficulty_function_values = [
            f"Difficulty Rule {int(i+1)} --> [{difficulty_function_rules[str(i+1)]}] with weight {float(w):.2f}" 
            for i, w in enumerate(difficulty_rules_weights)
        ]

    value_function_str = "\n============================================\n"
    value_function_str += "--- Value Rules ---\n"
    value_function_str += "\n".join(value_function_values)
    
    difficulty_function_str = "--- Difficulty Rules ---\n"
    difficulty_function_str += "\n".join(difficulty_function_values)
    difficulty_function_str += "\n============================================\n"

    ### Board coordinates
    board_coordinates_str = ""
    board_coordinates_str += f"Balls:\n"
    for k, v in state.ball_positions.items():
        if isinstance(v[0], str):
            continue
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"
    board_coordinates_str += f"Pockets:\n"
    for k, v in state.pocket_positions.items():
        board_coordinates_str += f"'{k}': ({v[0]:.2f},{v[1]:.2f})\n"

    ### Events
    events_str = ""
    events_str += f"Events:\n"
    for e in events:
        events_str += f"'{e.encoding}' at {e.pos}\n"

    ### Estimate weights
    response = dspy_explain_DEF(
        seed=f"{seed}",
        shot_params=shot_params_str,
        board_coordinates=board_coordinates_str,
        events=events_str,
        value_function_rules_vals=value_function_str,
        difficulty_function_rules_vals=difficulty_function_str,
    )
    return response

class EstimationTask:
    def __init__(self, entry, model_id, num_estimations=3):
        self.entry = entry
        self.model_id = model_id
        self.num_estimations = num_estimations
        self.state = State.from_json(entry['starting_state'])
        self.action = entry['params']
        self.assigned_llms = {}

class SequentialEstimationManager:
    def __init__(self, experiment_name, gpu_ids, gpu_size=40, max_count=50, num_estimations=3, reset=False):
        self.experiment_name = experiment_name
        self.gpu_size = gpu_size
        self.gpu_ids = gpu_ids
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.console = Console()
        self.max_count = max_count
        self.num_estimations = num_estimations
        self.reset = reset

        self.temperature = 0.05
        self.max_tokens = 1024
        self.repetition_penalty = 1.1
        
        # Initialize pool environment and function chooser
        self.env = Pool()
        self.chooser = FunctionChooser(target_balls=['red', 'blue', 'yellow'])
        
        # Setup logging
        self.log_dir = os.path.join(ROOT_DIR, "experiments", "experiment_two", "logs", self.timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        self.logger = self.setup_logging()
        
        # Load previous results and data
        self.estimations = self.load_previous_results()
        self.training_data = self.load_training_data()
        
        self.logger.info(f"Loaded previous estimations for {len(self.estimations)} models")
    
    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.log_dir}/main.log"),
                logging.StreamHandler()
            ]
        )
        return logging.getLogger()

    def load_previous_results(self):
        try:
            with open(f"{DATA_DIR}/exp2-dataset-scores.json", "r") as f:
                estimations = json.load(f)
                estimations["timestamp"] = datetime.now().isoformat()
        except FileNotFoundError:
            estimations = {"timestamp": datetime.now().isoformat()}
        return estimations

    def load_training_data(self):
        with open(f"{DATA_DIR}/skill_estimate_dataset.json", "r") as f:
            training_data = json.load(f)
        return training_data

    def get_model_size(self, model_id):
        match = re.search(r'-(\d+)b', model_id.lower())
        return int(match.group(1)) if match else 0
    
    def get_required_gpus(self, model_size):
        return 1 if model_size < MAX_MODEL_SIZE_FOR_SINGLE_GPU else (model_size - 1) // MAX_MODEL_SIZE_FOR_SINGLE_GPU + 1
    
    def is_api_model(self, model_id):
        return any([name in model_id for name in ['gpt', 'together']])

    def load_model(self, model_id, temperature=0.2, max_tokens=2048):
        """Load a model based on its type and requirements"""
        if self.is_api_model(model_id):
            return LLM(model_id, temperature=temperature, max_tokens=max_tokens, repetition_penalty=self.repetition_penalty)
        
        model_size = self.get_model_size(model_id)
        required_gpus = self.get_required_gpus(model_size)
        assigned_gpus = self.gpu_ids[:required_gpus]
        
        return LLM(model_id, assigned_gpus, self.gpu_size, repetition_penalty=self.repetition_penalty)

    def save_results(self, task_dir, estimation_data):
        results_file = os.path.join(task_dir, "results.json")
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                current_results = json.load(f)
        else:
            current_results = []
        current_results.append(estimation_data)
        with open(results_file, "w") as f:
            json.dump(current_results, f, indent=2)
            
        # Save to main estimations file
        with open(f"{DATA_DIR}/exp2-dataset-scores.json", "w") as f:
            json.dump(self.estimations, f, indent=2)

    @weave.op()
    def run_estimation_task(self, task):
        self.logger.info(f"Starting estimation task for model: {task.model_id}")
        
        # Load model for the task
        model = self.load_model(task.model_id, temperature=self.temperature, max_tokens=self.max_tokens)
        task.assigned_llms[task.model_id] = model
        
        try:
            # Initialize environment with task state
            self.env.from_state(task.state)
            self.env.strike(**task.action)
            events = self.env.get_events()
            end_state = self.env.get_state()
            
            # Calculate function values
            _, _, _, raw_values, raw_difficulties = self.chooser.evaluate_shots(
                task.state, 
                [task.action], 
                [events], 
                [end_state]
            )
            raw_values, raw_difficulties = raw_values[0], raw_difficulties[0]
            normalized_values, normalized_difficulties = self.chooser.normalise(
                raw_values, 
                raw_difficulties
            )
            
            # Prepare estimation data
            estimation_data = {
                "state": task.state.to_json(),
                "end_state": end_state.to_json(),
                "shot_params": task.action,
                "events": [e.to_json() for e in events],
                "raw_values": raw_values.tolist(),
                "raw_difficulty": raw_difficulties.tolist(),
                "normalized_values": normalized_values.tolist(),
                "normalized_difficulty": normalized_difficulties.tolist(),
                "estimations": {
                    "with_functions": [],
                    "without_functions": [],
                    "groundtruth_weights": {
                        "value_weights": normalized_values.tolist(),
                        "difficulty_weights": normalized_difficulties.tolist(),
                    },
                }
            }
            
            # Generate multiple estimations
            for _ in range(task.num_estimations):
                # With DEF
                withDEF_response = explain_DEF_with_DEF_fn(
                    model.llm,
                    task.state,
                    task.action,
                    events,
                    end_state,
                    normalized_values.tolist(),
                    normalized_difficulties.tolist(),
                    seed=np.random.randint(low=0,high=1000000),
                    percentage_values=True
                )
                
                value_expl = [
                    f"The applicability of value rule {ridx+1} is {getattr(withDEF_response, f'value{ridx+1}_est', 'moderate').value}."
                    for ridx in range(len(normalized_values))
                ]
                difficulty_expl = [
                    f"The applicability of difficulty rule {ridx+1} is {getattr(withDEF_response, f'difficulty{ridx+1}_est', 'moderate').value}."
                    for ridx in range(len(normalized_difficulties))
                ]
                
                rdict = {
                    'value_explanations': value_expl,
                    'difficulty_explanations': difficulty_expl,
                }
                estimation_data["estimations"]['with_functions'].append(rdict)
                
                # Without DEF
                withoutDEF_response = explain_DEF_without_DEF_fn(
                    model.llm,
                    task.state,
                    task.action,
                    events,
                    end_state,
                    seed=np.random.randint(low=0,high=1000000)
                )
                
                value_expl= [
                    f"The applicability of value rule {ridx+1} is {getattr(withoutDEF_response, f'value{ridx+1}_est', 'moderate').value}."
                    for ridx in range(len(normalized_values))
                ]
                difficulty_expl = [
                    f"The applicability of difficulty rule {ridx+1} is {getattr(withoutDEF_response, f'difficulty{ridx+1}_est', 'moderate').value}."
                    for ridx in range(len(normalized_difficulties))
                ]
                
                rdict = {
                    'value_explanations': value_expl,
                    'difficulty_explanations': difficulty_expl,
                }
                estimation_data["estimations"]['without_functions'].append(rdict)
            
            return estimation_data
            
        finally:
            # Clean up model
            model.delete()
            task.assigned_llms.pop(task.model_id, None)
        
    def print_state(self, completed_tasks, total_tasks):
        table = Table(title="Estimation Progress", box=box.ROUNDED)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Completed Tasks", f"{completed_tasks}/{total_tasks}")
        table.add_row("Progress", f"{(completed_tasks/total_tasks)*100:.1f}%")
        
        self.console.print(table)
        self.console.print("\n" + "="*50 + "\n")

    def run(self, model_ids):
        total_tasks = 0
        tasks = []
        
        # Create tasks for each model
        for model_id in model_ids:
            if self.reset or model_id not in self.estimations:
                self.estimations[model_id] = []
                
            existing_count = len(self.estimations[model_id])
            for entry in self.training_data[existing_count:self.max_count]:
                tasks.append(EstimationTask(entry, model_id, self.num_estimations))
                total_tasks += 1
        
        self.logger.info(f"Created {total_tasks} estimation tasks")
        
        # Run tasks sequentially
        for i, task in enumerate(tasks):
            try:
                model_key = task.model_id.split("/")[-1]
                task_dir = os.path.join(self.log_dir, "tasks", model_key)
                os.makedirs(task_dir, exist_ok=True)
                
                estimation_data = self.run_estimation_task(task)
                self.estimations[task.model_id].append(estimation_data)
                self.save_results(task_dir, estimation_data)
                
                self.print_state(i + 1, total_tasks)
                
            except Exception as e:
                self.logger.error(f"Error in task for model {task.model_id}: {str(e)}")
                self.logger.error(traceback.format_exc())
                continue

def run_estimation_experiment(experiments_config, args):
    # Get experiment size from environment
    if os.getenv("SIZE"):
        lm_size = os.getenv("SIZE")
        assert lm_size in ['SMALL', 'MEDIUM', 'LARGE'], "Invalid SIZE. Choose 'SMALL', 'MEDIUM', or 'LARGE'."
    else:
        lm_size = 'SMALL'
    
    # Select models based on type
    model_ids = []
    if args.model_type == "api":
        model_ids = experiments_config['models']['api']
    elif args.model_type == "together":
        model_ids = experiments_config['models']['together']['text']
    elif args.model_type == "local":
        model_ids = experiments_config['models']['local']['text'][lm_size]
    elif args.model_type == "custom":
        model_ids = experiments_config['models']['custom']
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")
        
    # Initialize experiment description for Weave
    exp_description = f"{args.model_type}_text_{lm_size}"
    weave.init(f"ExperimentTwo-{exp_description}")
        
    # Setup GPUs
    gpu_ids = [f"gpu_{i}" for i in args.gpu_ids.split(',') if ',' in args.gpu_ids]
    
    # Initialize manager
    manager = SequentialEstimationManager(
        "experiment_two",
        gpu_ids,
        gpu_size=args.gpu_size,
        max_count=args.max_count,
        num_estimations=args.k_estimations,
        reset=args.reset
    )
    
    # Run experiment
    manager.run(model_ids)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run estimation experiment")
    parser.add_argument("--model_type", choices=["api", "together", "local", "custom"], 
                      help="Choose model type",
                      default="together")
    parser.add_argument("--gpu_ids", type=str, default='0',
                      help="Comma-separated list of GPU IDs to use")
    parser.add_argument("--gpu_size", type=int, default=40,
                      help="Size of GPU in GB")
    parser.add_argument("--max_count", type=int, default=50,
                      help="Maximum number of shots to analyze")
    parser.add_argument("--k_estimations", type=int, default=3,
                      help="Number of estimations per shot")
    parser.add_argument("--reset", type=bool, default=False,
                      help="Whether to reset and regenerate all estimations")
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(ROOT_DIR, "experiments", "experiments_config.yaml") 
    with open(config_path) as f:
        experiments_config = yaml.safe_load(f)
        
    # Setup model access if needed
    if args.model_type in ['local']:
        from huggingface_hub import login
        assert "HUGGINGFACE_TOKEN" in os.environ
        login(os.environ["HUGGINGFACE_TOKEN"])

    if args.model_type in ['api', 'custom']:
        assert "OPENAI_KEY" in os.environ, "Please set OPENAI_KEY environment variable"

    if args.model_type in ['together', 'custom']:
        assert 'TOGETHER_API_KEY' in os.environ, "Please set TOGETHER_API_KEY environment variable"
        
    run_estimation_experiment(experiments_config, args)