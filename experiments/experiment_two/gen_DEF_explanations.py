import json, dspy, argparse, os
import numpy as np
import weave
import yaml

from typing import List
from datetime import datetime

from poolagent.path import DATA_DIR, ROOT_DIR
from poolagent.pool import Pool
from poolagent.agents import FunctionChooser
from poolagent.experiment_manager import ExperimentManager, Experiment, Task
from poolagent.utils import State, Event

from pydantic import BaseModel, Field
from enum import Enum

class EstimateEnum(str,Enum):
    very_low=           "very_low"
    low=                "low"
    moderately_low=     "moderately_low"
    moderate=           "moderate"
    moderately_high=    "moderately_high"
    high=               "high"
    very_high=          "very_high"
    unknown=            "unknown"

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

    You must return an estimate of the applicability of each of the value and difficulty rules.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a percentage where 0% means that the rule has very low applicability to the current state and shot, and 100% means that it has very high applicability.")
    #value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, with the weight as a scalar between [-1,+1] where a value close to -1 means that the rule has very low applicability to the current state and shot, and a value close to +1 means that it has very high applicability.")
    #difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, with the weight as a scalar between [-1,+1] where a value close to -1 means that the rule has very low applicability to the current state and shot, and a value close to +1 means that it has very high applicability.")

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


class ExplainDEFWithDEFSignature_v0(dspy.Signature):
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
        - The value rules and weights
        - The difficulty rules and weights

    You must return explanations for all of the value and difficulty rules.
    """
    seed = dspy.InputField(desc="The seed for the explanation, you can ignore this.")
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. between [-1,+1] where a value close to +1 means the rule applies to the current state and shot.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. between [-1,+1] where a value close to +1 means the shot is hard for the reason stated in the rule.")

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
    dspy_explain_DEF = dspy.ChainOfThought(ExplainDEFWithOutDEFSignature)

    shot_params_str = ""
    for key, value in shot_params.items():
        shot_params_str += f"{key}: {value:.2f}\n"

    ### Load Rules and collect the top 3 and bottom 3 rules
    value_function_rules = {}
    with open(f"{DATA_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    difficulty_function_rules = {}
    with open(f"{DATA_DIR}/difficulty_rules.json") as f:
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
        dspy_explain_DEF = dspy.ChainOfThought(ExplainDEFWithDEFPercentageSignature)
    else:
        dspy_explain_DEF = dspy.ChainOfThought(ExplainDEFWithDEFSignature)
    
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
    with open(f"{DATA_DIR}/value_rules.json") as f:
        value_function_rules = json.load(f)
    difficulty_function_rules = {}
    with open(f"{DATA_DIR}/difficulty_rules.json") as f:
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


class ExplanationTask(Task):
    def __init__(self, entry, model):
        super().__init__(description=f"Model=[{model}]", models=[model])
        self.entry = entry
        self.state = entry['state']
        self.action = entry['action']

class ExplanationExperiment(Experiment):
    def __init__(
        self, 
        llm_models, 
        max_concurrent_threads, 
        max_count=50, 
        k_estimations=3, 
        percentage_values=True,
        reset=False,
    ):
        '''
        :param reset: bool that specifies whether to create explanations from scratch or 
            simply pickup from where we left it off last time.
        '''
        super().__init__()
        self.llm_models = llm_models
        self.max_count = max_count
        self.k_estimations = k_estimations
        self.percentage_values = percentage_values
        self.reset = reset
        self.envs = [Pool() for _ in range(max_concurrent_threads)]
        self.chooser = FunctionChooser(target_balls=['red', 'blue', 'yellow'])
        self.load_data()
        self.create_tasks()

    def load_data(self):
        with open(f"{DATA_DIR}/stochastic_training_data.json", "r") as f:
            training_data = json.load(f)['train']
        shuffle_indices = np.random.permutation(int(self.max_count * 1.5))
        self.training_data = [training_data[i] for i in shuffle_indices]

        try:
            output_path = f"{DATA_DIR}/DEF_explanations-k{self.k_estimations}-n{self.max_count}-{'percentages' if self.percentage_values else 'scalars'}.json"
            with open(output_path, "r") as f:
                self.explanations = json.load(f)
                self.explanations["timestamp"] = datetime.now().isoformat()
        except FileNotFoundError:
            self.explanations = {"timestamp": datetime.now().isoformat()}

    def create_tasks(self):
        for model in self.llm_models:
            #if '/' in model:
            #    model = model.split('/')[-1]
            
            if self.reset \
            or model not in self.explanations:
                self.explanations[model] = []

            existing_count = len(self.explanations[model])

            for entry in self.training_data[existing_count:self.max_count]:
                self.tasks.append(ExplanationTask(entry, model))

    def save_results(self, results_dir, explanation_data):
        os.makedirs(results_dir, exist_ok=True)
        results_file = os.path.join(
            results_dir, 
            f"results-k{self.k_estimations}-n{self.max_count}-{'percentages' if self.percentage_values else 'scalars'}.json",
        )

        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                current_results = json.load(f)
        else:
            current_results = []

        current_results.append(explanation_data)

        with open(results_file, "w") as f:
            json.dump(current_results, f, indent=2)

    def run_task(
        self, 
        thread_id, 
        task, 
        timestamp, 
        N=1, 
        logger=None, 
        use_wandb=True,
        percentage_values=False,
        **kwargs,
    ):
        model = task.models[0]
        llm = task.assigned_llms[model]
        '''
        if hasattr(llm.llm, 'model'):
            model_id = llm.llm.model
        else:
            model_id = llm.llm.kwargs['model']
        '''
        model_id = model
        if use_wandb:
            print(model_id)
            weave.init(f"SimLM-DEF-explanation-{model_id.replace('/','-')}") 
        
        self.current_tasks[f"{task.description}_thread{thread_id}"] = [0]
        if logger:
            logger.info(f"Thread ID: {thread_id}, Number of games: {N}")
            logger.info(f"Starting explanation generation for model {model}")

        start_state = State.from_json(task.state)
        action = task.action

        if logger:
            logger.info(f"Starting state: {task.state}")
            logger.info(f"Action: {task.action}")

        env = self.envs[thread_id]
        env.from_state(start_state)
        env.strike(**action)
        events = env.get_events()
        end_state = env.get_state()

        if logger:
            logger.info(f"Events: {[e.to_json() for e in events]}")
            logger.info(f"End state: {end_state.to_json()}")

        _, _, _, raw_values, raw_difficulties = self.chooser.evaluate_shots(start_state, [action], [events], [end_state])
        raw_values, raw_difficulties = raw_values[0], raw_difficulties[0]
        normalized_values, normalized_difficulties = self.chooser.normalise(
            raw_values, 
            raw_difficulties,
            percentages=percentage_values,
        )

        if logger:
            logger.info(f"Raw values: {raw_values.tolist()}")
            logger.info(f"Raw difficulties: {raw_difficulties.tolist()}")
            logger.info(f"Normalized values: {normalized_values.tolist()}")
            logger.info(f"Normalized difficulties: {normalized_difficulties.tolist()}")

        explanation_data = {
            "state": start_state.to_json(),
            "end_state": end_state.to_json(),
            "shot_params": action,
            "events": [e.to_json() for e in events],
            "raw_values": raw_values.tolist(),
            "raw_difficulty": raw_difficulties.tolist(),
            "normalized_values": normalized_values.tolist(),
            "normalized_difficulty": normalized_difficulties.tolist(),
        }

        @weave.op()
        def generate_explanations(
            starting_state, 
            shot_params, 
            events, 
            end_state, 
            norm_values, 
            norm_difficulty,
            percentage_values,
        ):
            seeds = np.random.randint(0, 10000, self.k_estimations)
            llm_dspy_object = llm.llm
            odict = {
                "with_functions": [],
                "without_functions": [],
            }

            for i in range(self.k_estimations):
                try:
                    withDEF_response = explain_DEF_with_DEF_fn(
                        llm_dspy_object, 
                        starting_state, 
                        shot_params, 
                        events, 
                        end_state, 
                        norm_values, 
                        norm_difficulty, 
                        seed=seeds[i],
                        percentage_values=percentage_values,
                    )
                    value_expl = [
                        #getattr(withDEF_response, f'value{ridx+1}_expl')
                        f"The applicability of value rule {ridx+1} is {getattr(withDEF_response, f'value{ridx+1}_est').value}."
                        for ridx in range(len(norm_values))
                    ]
                    difficulty_expl = [
                        #getattr(withDEF_response, f'difficulty{ridx+1}_expl')
                        f"The applicability of difficulty rule {ridx+1} is {getattr(withDEF_response, f'difficulty{ridx+1}_est').value}."
                        for ridx in range(len(norm_difficulty))
                    ]
                except Exception as e:
                    print(e)
                    value_expl = [
                        f"The applicability of value rule {ridx+1} is UNKNOWN."
                        for ridx in range(len(norm_values))
                    ]
                    difficulty_expl = [
                        f"The applicability of difficulty rule {ridx+1} is UNKNOWN."
                        for ridx in range(len(norm_difficulty))
                    ]

                rdict = {
                    'value_explanations': value_expl,
                    'difficulty_explanations': difficulty_expl,
                }
                odict['with_functions'].append(rdict)
                
                try:
                    withoutDEF_response = explain_DEF_without_DEF_fn(
                        llm_dspy_object, 
                        starting_state, 
                        shot_params, 
                        events, 
                        end_state, 
                        seed=seeds[i],
                        percentage_values=percentage_values,
                    )
                    value_expl= [
                        #getattr(withoutDEF_response, f'value{ridx+1}_expl')
                        f"The applicability of value rule {ridx+1} is {getattr(withoutDEF_response, f'value{ridx+1}_est').value}."
                        for ridx in range(len(norm_values))
                    ]
                    difficulty_expl = [
                        #getattr(withoutDEF_response, f'difficulty{ridx+1}_expl')
                        f"The applicability of difficulty rule {ridx+1} is {getattr(withoutDEF_response, f'difficulty{ridx+1}_est').value}."
                        for ridx in range(len(norm_difficulty))
                    ]
                except Exception as e:
                    print(e)
                    value_expl = [
                        f"The applicability of value rule {ridx+1} is UNKNOWN."
                        for ridx in range(len(norm_values))
                    ]
                    difficulty_expl = [
                        f"The applicability of difficulty rule {ridx+1} is UNKNOWN."
                        for ridx in range(len(norm_difficulty))
                    ]

                rdict = {
                    'value_explanations': value_expl,
                    'difficulty_explanations': difficulty_expl,
                }
                odict['without_functions'].append(rdict)
            return odict

        explanation_data["explanations"] = generate_explanations(
            start_state,
            action,
            events,
            end_state,
            normalized_values.tolist(),
            normalized_difficulties.tolist(),
            percentage_values=percentage_values,
        )
        self.explanations[model].append(explanation_data)

        if logger:
            logger.info(f"Generated explanations for model {model}, entry {len(self.explanations[model])}")

        results_dir = f"{ROOT_DIR}/experiments/experiment_two/DEF_explanations_logs/{timestamp}/tasks/{task.description}/"
        self.save_results(results_dir, explanation_data)

        del self.current_tasks[f"{task.description}_thread{thread_id}"]

        logger.info(f"Completed explanation generation for model {model}, entry {len(self.explanations[model])}")


def generate_dataset(
        experiments_config,
        model_type, 
        max_count, 
        k_estimations, 
        gpu_ids='0', 
        n_threads=3,
        temperature=0.0,
        max_tokens=4096,
        reset=False,
        percentage_values=False,
    ):
    '''
    :param reset: bool specifying whether the explanations should be generated anew from scratch
        or whether to restart from where it was left off previously.
    '''
    # Define models based on the chosen type
    assert model_type in ["api", "local"], "Invalid model type. Choose 'api' or 'local'."
    if os.getenv("SIZE"):
        lm_size_for_experiment = os.getenv("SIZE")
        assert lm_size_for_experiment in ['SMALL', 'MEDIUM', 'LARGE'], "Invalid experiment SIZE variable. Choose 'SMALL', 'MEDIUM', or 'LARGE'."
    else:
        lm_size_for_experiment = 'SMALL'

    api_model_ids = experiments_config['models']['api'] 
    text_model_ids = experiments_config['models']['local']['text'][lm_size_for_experiment]
    vision_model_ids = experiments_config['models']['local']['vision'][lm_size_for_experiment]

    if model_type == 'local':
        model_ids = text_model_ids
    else:
        model_ids = api_model_ids

    #gpu_ids = [f"gpu_{i}" for i in range(1, num_gpus+1)]
    # TODO: figure out why guarding for ',' ? 
    #gpu_ids = [f"gpu_{i}" for i in gpu_ids.split(',') if ',' in gpu_ids]
    gpu_ids = [f"gpu_{i}" for i in gpu_ids.split(',') ]

    # Initialize ExpManager
    experiment = ExplanationExperiment(
        model_ids, 
        max_concurrent_threads=n_threads, 
        max_count=max_count, 
        k_estimations=k_estimations,
        percentage_values=percentage_values,
        reset=reset,
    )
    exp_manager = ExperimentManager(
        "experiment_two_DEF_explanations", 
        model_ids, 
        gpu_ids, 
        experiment, 
        max_concurrent_threads=n_threads,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    # Run the experiment
    task_kwargs = {
        #'explicit': explicit,
        'percentage_values': percentage_values,
    }
    exp_manager.run(**task_kwargs)

    # Once finished, copy the final set of results to the data directory
    output_path = f"{DATA_DIR}/DEF_explanations-k{k_estimations}-n{max_count}-{'percentages' if percentage_values else 'scalars'}.json"
    with open(output_path, "w") as f:
        json.dump(experiment.explanations, f, indent=2)


def str2bool(inp):
    if not isinstance(inp, bool):
        assert isinstance(inp, str)
        inp = 'true' in inp.lower()
    return inp


if __name__ == "__main__":
    from huggingface_hub import login
    assert "HUGGINGFACE_TOKEN" in os.environ, "Please set the Hugging Face token as an environment variable"
    HF_TOKEN = os.environ["HUGGINGFACE_TOKEN"]
    login(HF_TOKEN)

    parser = argparse.ArgumentParser(description="Run simulation with API or local models")
    parser.add_argument("model_type", choices=["api", "local"], help="Choose 'api' for GPT models or 'local' for Hugging Face models")
    parser.add_argument("--gpu_ids", type=str, default='0', help="Ids of GPUs to use")
    parser.add_argument("--n_threads", type=int, default=3, help="Number of concurrent threads")
    parser.add_argument("--max_count", type=int, default=50, help="Number of shots to explain")
    parser.add_argument("--k_estimations", type=int, default=3, help="Number of estimations to generate per shot")
    parser.add_argument("--temperature", type=float, default=0.0, help="Temperature to use LLMs with.")
    parser.add_argument("--max_tokens", type=int, default=4096, help="Max number of tokens to use when generating tokens with LLMs.")
    #parser.add_argument("--explicit", type=str2bool, default=False, help="Whether to ask the LM for ach DEF weights explicitly, one at a time, or to ask for a list.")
    parser.add_argument("--percentage_values", type=str2bool, default=False, help="Whether to display value function outputs as percentages or as scalars between [-1,1].")
    parser.add_argument("--reset", type=str2bool, default=False, help="Whether to restart generation from scratch.")
    parser.add_argument("--config", type=str, default="./experiments_config.yaml", help="YAML config file path.")
    args = parser.parse_args()

    experiments_config = data = yaml.safe_load(open(args.config))
    
    generate_dataset(
        experiments_config=experiments_config,
        model_type=args.model_type, 
        max_count=args.max_count, 
        k_estimations=args.k_estimations, 
        gpu_ids=args.gpu_ids, 
        n_threads=args.n_threads,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        #explicit=args.explicit,
        percentage_values=args.percentage_values,
        reset=args.reset,
    )

