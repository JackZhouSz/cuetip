import dspy
import logging
import os
from datetime import datetime
from poolagent.utils import State, Event


class ChooseShotSignature(dspy.Signature):
    """You will be presented with a list of shots to make in a game of pool/billiards. You must choose one of these shots to make, specified by its number in the list. 

    The IDs of the pockets are: left top (lt), right top (rt), left center (lc), right center (rc), left bottom (lb), right bottom (rb).
    
    The shots will be represented by the events that occured during the shot, with the format "BALL-BALL-X-Y" for a collision between two balls X and Y, "BALL-POCKET-X-Z" for a ball X falling into pocket Z, and "BALL-CUSHION-X" for a ball X colliding with a cushion (note there is no second argument needed). The shots will be presented as a numbered list, for example
    1. BALL-BALL-cue-blue, BALL-POCKET-blue-lb
    2. BALL-BALL-cue-red, BALL-CUSHION-red, BALL-POCKET-red-rc 
    3. ... 
    N. BALL-CUSHION-cue, BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt
    You must return the number of the shot you wish to make, from 1 to N.

    Try to reason about what makes a good shot in pool and why the chosen shot is a good one.
    """

    target_balls = dspy.InputField(desc="The IDs of the balls that you must pot to win.")
    shots = dspy.InputField(desc="The shots to choose from.")

    chosen_shot = dspy.OutputField(desc="The number of the shot to make, from 1 to N.")

class DEFChooseShotSignature(dspy.Signature):
    """You will be presented with a list of shots to make in a game of pool/billiards. You must choose one of these shots to make, specified by its number in the list. 

    The IDs of the pockets are: left top (lt), right top (rt), left center (lc), right center (rc), left bottom (lb), right bottom (rb).
    
    The shots will be represented by the events that occured during the shot, with the format "BALL-BALL-X-Y" for a collision between two balls X and Y, "BALL-POCKET-X-Z" for a ball X falling into pocket Z, and "BALL-CUSHION-X" for a ball X colliding with a cushion (note there is no second argument needed). The shots will be presented as a numbered list, for example
    1. BALL-BALL-cue-blue, BALL-POCKET-blue-lb
    2. BALL-BALL-cue-red, BALL-CUSHION-red, BALL-POCKET-red-rc 
    3. ... 
    N. BALL-CUSHION-cue, BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt
    You must return the number of the shot you wish to make, from 1 to N.

    You will also be provided with a list of values for each shot, which correspond to the 'Value Rules' and 'Difficulty Rules' shown below. If a value for a shot is high, then then corresponding rule applies to the shot, if the value is low, then the rule does not apply. The rules are as follows:
    
    **Value Rules**:
        1. Ball Groupings: Identify sets of two or more balls of the same type in close proximity that can be easily pocketed in sequence.
        2. Makable Regions: Assess areas on the table where balls can be pocketed without using kick or bank shots.
        3. Insurance Balls: Locate balls that can be easily pocketed from almost anywhere on the table.
        4. Break-up Opportunities: Evaluate clusters of balls that need separation.
        5. Safety Opportunities: Identify chances to play defensive shots that leave the opponent in a difficult position.
        6. Two-way Shot Possibilities: Look for shots that offer both offensive and defensive potential. 
        7. Table Layout for Safeties: Assess the overall layout for defensive play.
        8. Multiple-ball Positions: Consider the arrangement of multiple balls that need to be pocketed in sequence.
        9. Avoidance Shots: Recognize balls that should be avoided to maintain a favorable layout or to leave the opponent in a difficult position.
        10. Combination and Bank Shot Opportunities: While often more difficult, the presence of viable combination or bank shots can add value to a table state by providing additional options.
        11. Rail Proximity: Consider the position of balls near rails. 
        12. Scratch Potential: Evaluate the layout for potential scratches (fouls).
        13. Above all, prioritise shots that pot the most target balls.

    **Difficulty Rules**:
        1. Distance: Shot difficulty increases with greater distances between the cue ball, object ball, and pocket.
        2. Cut Angle: Larger cut angles are more challenging than straight or small angle shots. 
        3. Obstacle Balls: The presence of other balls obstructing the path of the cue ball or object ball significantly increases shot difficulty. 
        4. Rail Contact: Shots requiring the cue ball to hit a rail first (like rail cut shots) are more complex due to the need to account for rail dynamics and potential throw effects.
        5. English Requirements: Shots needing sidespin (English) are more difficult to control and execute.
        6. Speed Control: Shots requiring precise speed control, whether very fast or very slow, are more challenging. 
        7. Follow/Draw Needs: Shots requiring significant follow or draw are more difficult than natural roll shots.
        8. Rail Proximity: Balls very close to rails can be more difficult to hit cleanly and may require specialized techniques like rail cut shots or massé.
        9. Scratch Potential: Positions with a high risk of scratching are more difficult to play safely and effectively.
        10. Massé (Curve) Shots: Difficulty increases with the amount of curve required.
        11. Frozen Ball Situations: Balls touching each other or touching a rail create unique challenges.
        12. Multiple Effects: Shots involving a combination of factors (e.g., cut angle, speed, and English) are particularly challenging due to the need to account for multiple variables simultaneously.
        13. Throw Effects: Accounting for throw (both cut-induced and English-induced) adds complexity.
        14. Deflection and Cue Ball Curve: When using English, especially at higher speeds or with an elevated cue, accounting for cue ball deflection and curve increases shot difficulty.
        15. Multi-ball Collision: It is exponentially difficult to pot a ball by colliding it with multiple balls.
        16. Multi-cushion Collision: It is exponentially difficult to pot a ball by having it bounce off multiple cushions.

    Try to reason about what makes a good shot in pool and why the chosen shot is a good one.
    """
    target_balls = dspy.InputField(desc="The IDs of the balls that you must pot to win.")
    shots = dspy.InputField(desc="The shots to choose from.")
    value_rules = dspy.InputField(desc="The values that correspond to the value rules i.e. index 0 corresponds to rule 1, index 1 corresponds to rule 2, etc.")
    difficulty_rules = dspy.InputField(desc="The values that correspond to the difficulty rules i.e. index 0 corresponds to rule 1, index 1 corresponds to rule 2, etc.")

    chosen_shot = dspy.OutputField(desc="The number of the shot to make, from 1 to N.")

class LM_Chooser():

    def __init__(self, target_balls, defs=False):
        self.target_balls_str = ", ".join(target_balls)
        self.record = {}
        self.defs = defs
        if self.defs:
            self.choose_shot = dspy.ChainOfThought(DEFChooseShotSignature)
        else:
            self.choose_shot = dspy.ChainOfThought(ChooseShotSignature) 

    def parse_chosen_shot(self, chosen_shot, N, logger=None):
        """Returns the parsed chosen shot. If there is an error with the input, try to find the number in the text the appears first. Otherwise return -1.

        Args:
            chosen_shot (_type_): Text input from the LM.
            N (_type_): Max number of shots to choose from.

        Returns:
            _type_: The chosen shot.
        """
        if logger:
            logger.debug(f"Parsing chosen shot: {chosen_shot}, N: {N}")
        try:
            parsed_shot = int(chosen_shot)
            if logger:
                logger.info(f"Successfully parsed chosen shot as integer: {parsed_shot}")
            return parsed_shot
        except ValueError:
            if logger:
                logger.warning(f"Failed to parse chosen shot as integer: {chosen_shot}")
            
            n_present = []

            for i in range(1, N+1):
                if f"{i}" in chosen_shot.lower():
                    n_present.append(
                        (i, chosen_shot.lower().index(f"{i}"))
                    )

            if len(n_present) == 0:
                if logger:
                    logger.error("No valid shot number found in the chosen shot text")
                return -1
            
            n_present.sort(key=lambda x: x[1])
            if logger:
                logger.info(f"Extracted shot number from text: {n_present[0][0]}")
            return n_present[0][0]

    def __call__(self, shot_events, lm, logger=None, def_values=None):
        
        ### Setup the shot events
        shot_events_str = ""
        for i, shot in enumerate(shot_events):

            for e_idx, e in enumerate(shot):
                if 'cushion' in e:
                    shot[e_idx] = '-'.join(e.split('-')[:3])
                    
            shot_events_str += f"{i+1}. {', '.join([e.encoding for e in shot])}\n"

        args_config = {
            "stop": ['Target Balls:', '\n\n']
        }
        if logger:
            logger.info(f"Calling choose_shot with target_balls: {self.target_balls_str}")
        dspy.settings.configure(lm=lm)

        if self.defs and def_values:
            response = self.choose_shot(
                target_balls=self.target_balls_str,
                shots=shot_events_str,
                value_rules=def_values['value_rules'],
                difficulty_rules=def_values['difficulty_rules'],
                config=args_config,
                lm=lm
            )
        else:
            response = self.choose_shot(
                target_balls=self.target_balls_str,
                shots=shot_events_str,
                config=args_config,
                lm=lm
            )
        if logger:
            logger.info(f"Response from choose_shot: {response}")

        chosen_shot = self.parse_chosen_shot(response.chosen_shot, len(shot_events), logger) - 1

        self.record = {
            "target_balls": self.target_balls_str,
            "shots": shot_events_str,
            "response": {
                "reasoning": response.rationale,
                "chosen_shot": response.chosen_shot
            },
            "chosen_shot": chosen_shot
        }

        if chosen_shot < 0 or chosen_shot >= len(shot_events):
            if logger:
                logger.error(f"Invalid chosen shot: {chosen_shot}. Must be between 0 and {len(shot_events)-1}.")
            raise ValueError(f"Chosen shot must be between 1 and the number of shots, got {chosen_shot}.")
        if logger: 
            logger.info(f"Chosen shot: {chosen_shot}")
        return chosen_shot
