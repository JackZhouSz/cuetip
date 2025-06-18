import dspy

class FullExplainSingleShotSignature(dspy.Signature):
    """You are tasked with explaining why a particular shot in a game of pool is better than the alternatives. You are provided with the events that occurred in each shot, the best shot, and the pros and cons of each shot. You must return an explanation of the choice of the best shot, with regards to the other shots in the list. Make sure to emphasise the best shot in the explanation. 
    
    The events are given in the format:
        - BALL-BALL-X-Y (meaning a collision between balls X and Y), 
        - BALL-CUSHION-X (meaning a collision between ball X and a cushion), 
        - BALL-POCKET-X-Z (meaning ball X fell into pocket Z).

    The Pro and Con of each shot is shown as a 'Rule' given by an expert about playing pool. A rule given as a 'Pro' is a positive aspect of the shot, i.e. high value, and a rule given as a 'Con' is a negative aspect of the shot, i.e. high difficulty. The chosen shot is the most valuable and the least difficult, so make sure to emphasise this in the explanation.

    Create an explanation that utilises the most important information and be concise and informative. Make sure to explain why the best shot is the best, and why the other shots are not as good. Be sure to explain why each alternative shot is not as good as the chosen shot, but also be sure to mention their good qualities too, with respect to the rules, and the events that occurred in the shot. Keep in mind that these are potential shots for a user to make, so make your explanation as if you're explaining to a curious student who wants to learn more about the game of pool. It is very important that you do not make up information, i.e. only say things that are verifiably true.
    """

    events = dspy.InputField(desc="The events that occurred in each shot")
    best_shot_index = dspy.InputField(desc="The best shot, make sure that this shot is emphasised in the explanation")
    pros_cons = dspy.InputField(desc="The pro and con of each shot")

    explanation = dspy.OutputField(desc="The explanation of the choice of the best shot, with regards to the other shots.")

class FullUnconditionalExplainSingleShotSignature(dspy.Signature):
    """You are tasked with explaining why a particular shot in a game of pool is better than the alternatives. You are provided with the events that occurred in each shot and the number of the best shot. You must return an explanation of the choice of the best shot, with regards to the other shots in the list. Make sure to emphasise the best shot in the explanation. 
    
    The events are given in the format:
        - BALL-BALL-X-Y (meaning a collision between balls X and Y), 
        - BALL-CUSHION-X (meaning a collision between ball X and a cushion), 
        - BALL-POCKET-X-Z (meaning ball X fell into pocket Z).

    Create an explanation that utilises the most important information and be concise and informative. Make sure to explain why the best shot is the best, and why the other shots are not as good. Be sure to explain why each alternative shot is not as good as the chosen shot, but also be sure to mention their good qualities too, with respect to the events that occurred in the shot. Keep in mind that these are potential shots for a user to make, so make your explanation as if you're explaining to a curious student who wants to learn more about the game of pool. It is very important that you do not make up information, i.e. only say things that are verifiably true.
    """

    events = dspy.InputField(desc="The events that occurred in each shot")
    best_shot_index = dspy.InputField(desc="The best shot, make sure that this shot is emphasised in the explanation")

    explanation = dspy.OutputField(desc="The explanation of the choice of the best shot, with regards to the other shots.")


class ChooseBetweenExplanations(dspy.Signature):
    """You are shown two explanations of a shot made in a game of pool/billiards, and you must choose which one is better. The explanations are provided in the format: 'Explanation 1: ... Explanation 2: ...'. The better explanation is the one that is more informative and provides a better understanding of the pros and cons of the shot. You are also shown the events and parameters of the shot, so that if an explanation is missing important information you can choose the other one, this is very important to check. To make a decision simply return either ONE or TWO, depending on which explanation you think is better. DO NOT return both ONE and TWO in a single response, they are mutually exclusive."""

    shot_params = dspy.InputField(desc="The parameters of the shot, in the format V0, theta, phi, a, b.")
    shot_events = dspy.InputField(desc="The events of the shot, in the format BALL-BALL-X-Y (meaning a collision between balls X and Y), BALL-CUSHION-X (meaning a collision between ball X and a cushion), BALL-POCKET-X-Z (meaning ball X fell into pocket Z).")
    explanation1 = dspy.InputField(desc="The first explanation of the shot.")
    explanation2 = dspy.InputField(desc="The second explanation of the shot.")

    better_explanation = dspy.OutputField(desc="The better explanation of the shot.")

class EventsToDescription(dspy.Signature):
    """You are provided with a list of events that have occurred in a shot of pool. You must turn this series of events into a simple and easily interpretable description of the shot that was taken. For example, make sure to include the balls that were hit, when important balls contacted a cushion, and any balls that were pocketed."""

    events_str = dspy.InputField(desc="A list of events that occurred in the shot, in the format BALL-BALL-X-Y (meaning a collision between balls X and Y), BALL-CUSHION-X (meaning a collision between ball X and a cushion), BALL-POCKET-X-Z (meaning ball X fell into pocket Z).")
    pockets = dspy.InputField(desc="The pockets on the table.")

    description = dspy.OutputField(desc="A simple and easily interpretable description of the shot that was taken.")

class ExplainSingleShotSignature(dspy.Signature):
    """You are tasked with explaining the pros and cons of a particular shot in a game of pool using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow', and not 'green', 'black', or 'pink'
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

    You must return an explanation of the pros and cons of the shot, that takes into account both the value and difficulty of the shot, with regard to the rules provided. You are also given how each rule applies to the current state and shot, as a statement:
        - None
        - Low 
        - Medium
        - High
        - Extremely high
    Imagine you are explaining to a curious student who wants to learn more about the game of pool. Make it seem natural and conversational, while also thorough and full of detail, and be sure to not refer to numbers of the rules and weights, you must rewrite them into something more natural. Also, be sure to explain any pool specific words used. Above all, keep the explanation short and concise, no more than 10 lines."""
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")
    value_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the value of the shot, i.e. a high value means the shot is good for the reason stated in the rule, and a low value means the shot is bad for that reason.")
    difficulty_function_rules_vals = dspy.InputField(desc="The rules and weights that contribute to the difficulty of the shot, i.e. a high difficulty means the shot is hard for the reason stated in the rule, and a low value means the shot is easy for that reason.")

    explanation = dspy.OutputField(desc="the explanation of the shot that takes into account both the value and difficulty of the shot and makes reference to the rules and weights that contribute to the value and difficulty of the shot.")

class ExplainSingleShotSignatureNoFunctions(dspy.Signature):
    """You are tasked with explaining the pros and cons of a particular shot in a game of pool using the following information:
        - The target balls, i.e. the balls that should be potted, are 'blue', 'red', and 'yellow', and not 'green', 'black', or 'pink'        
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

    You must return an explanation of the pros and cons of the shot, that takes into account both the value and difficulty of the shot, with regard to the rules provided. Imagine you are explaining to a curious student who wants to learn more about the game of pool. Make it seem natural and conversational, while also thorough and full of detail, and be sure to not refer to numbers of the rules, you must rewrite them into something more natural. Also, be sure to explain any pool specific words used. Above all, keep the explanation short and concise, no more than 10 lines."""
    shot_params = dspy.InputField(desc="Shot parameters")
    board_coordinates = dspy.InputField(desc="The exact (x,y) coordinates of each ball and pocket on the table")
    events = dspy.InputField(desc="The events that occurred in the shot, and their positions")

    explanation = dspy.OutputField(desc="the explanation of the shot that takes into account both the value and difficulty of the shot.") 

class ActSignature(dspy.Signature):
    """You are a helpful AI Pool playing assistant. Reply to the user with an answer to their query. This is done by finding shots on a pool table that match with the users request. The included image depicts the current state of the table, use it to inform your decision."""
    conversation = dspy.InputField(desc="the conversation between user and assistant")
    current_balls = dspy.InputField(desc="the balls currently on the table")
    pockets = dspy.InputField(desc="the pockets on the table")
    previous_shots = dspy.InputField(desc="previous shots that have been attempted in this position, DO NOT repeat them.")

    message = dspy.OutputField(desc="message the user back with a suggested shot to take", prefix="Assistant: ")

class MakeShotSignature(dspy.Signature):
    """Describe a shot on a pool table that fulfills the provided goal. The included image depicts the current state of the table, use it to inform your decision. Describe the shot using the events that you wish to occur, using the notation: "BALL-BALL-X-Y" for a collision between two balls X and Y, "BALL-CUSHION-X" for a ball X colliding with a cushion, "BALL-POCKET-X-Z" for a ball X falling into pocket Z. You must also return a sub goal for the current shot, that is a description of what the shot will achieve that should build on previous shots to attempt to achieve the overall goal."""
    goals = dspy.InputField(desc="your overall plan and the sub goals of the previous shots", prefix="")
    current_balls = dspy.InputField(desc="the balls currently on the table, if a ball is not here then it has been potted.")
    target_balls = dspy.InputField(desc="the balls that you must attempt to pot.")
    pockets = dspy.InputField(desc="the pockets on the table.")
    previous_shots = dspy.InputField(desc="the previously attempted shots from this state, make sure to attempt new shots to find one that works.", prefix="Already Attempted: ")

    take_shot = dspy.OutputField(desc="whether you can take the shot, i.e. if the goal has already been completed then return END, otherwise return CONTINUE", prefix="Take Shot: ")
    sub_goal = dspy.OutputField(desc="the sub goal of the current shot, make sure it continues the overall plan.", prefix="Current Sub Goal: ")
    events = dspy.OutputField(desc="a comma separated list of events: BALL-BALL-X-Y, BALL-CUSHION-Y, BALL-POCKET-Y-Z, e.g. [BALL-BALL-cue-blue, BALL-POCKET-blue-lb] and [BALL-BALL-cue-red, BALL-CUSHION-red, BALL-POCKET-red-rc], etc", prefix="Events: ")
