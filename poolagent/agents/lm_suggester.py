import dspy
import logging
import os
from datetime import datetime
from poolagent.utils import dspy_setup, State, Event

class MultiShotSignatureImage(dspy.Signature):
    """You are tasked with suggesting shots to make in a game of pool/billiards. Based on the message you recieve and the current state of the pool table, you must suggest shots to make that satisfy the users goal. The included image depicts the current state of the table, use it to inform your decision. 

    The IDs of the pockets are: left top (lt) at (0,2), right top (rt) at (1,2), left center (lc) at (0,1), right center (rc) at (1,1), left bottom (lb) at (0,0), right bottom (rb) at (1,0).

    Try to observe in the image where the balls are on the board, and imagine which balls may be obstacles to potting the target balls. Try to work around them.
    
    The description of a shot is created using the events that you wish to occur, using the notation: "BALL-BALL-X-Y" for a collision between two balls X and Y, "BALL-POCKET-X-Z" for a ball X falling into pocket Z, and "BALL-CUSHION-X" for a ball X colliding with a cushion (note there is no second argument needed). Output the events of each shot as a comma separated list, with each shot on a new numbered line, for example:
    1. BALL-BALL-cue-blue, BALL-POCKET-blue-lb
    2. BALL-BALL-cue-red, BALL-CUSHION-red, BALL-POCKET-red-rc 
    3. ... 
    N. BALL-CUSHION-cue, BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt

    Try to be creative in your choice of events, but attempt to choose a shot that is simple and easy to execute. Make sure you do not repeat shots. Make sure a suggested shot does not foul by potting a ball that is not a target ball, hitting a non-target ball first, or by potting the cue ball.
    
    Use the Reasoning field to briefly think of what makes a good pool shot, and why you are choosing the shot you choose.
    """
    #    --- START EXAMPLE ---

    # Ball positions:
    #     Ball cue: (0.69, 0.29)
    #     Ball yellow: (0.42, 0.78)
    #     Ball black: (0.75, 1.50)
    #     Ball pink: (0.43, 1.54)

    # Target Balls: red, blue, yellow

    # Message: Suggest 3 shots to make in this position.

    # Number of shots: 3

    # Reasoning:
    #     The only available target ball is the yellow ball. We must plan three shots that attempt to pot this ball in different ways. The yellow ball is near the left center lc pocket, and the cue ball is positioned so that the yellow ball is almost between the cue ball and the pocket. One shot would be to simply knock the yellow ball into the lc pocket. Another shot could attempt the further pocket, left top lt, as the cue ball and is also well alligned for this pocket. The black ball is between the yellow ball and the right top rt pocket so that is not a good option. Lastly, we could try to bounce the cue ball off the right cushion and into the yellow ball, potting it in the lc pocket.

    # Shots:
    #     1. BALL-BALL-cue-yellow, BALL-POCKET-yellow-lc
    #     2. BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt
    #     3. BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lc

    # --- END EXAMPLE ---

    ball_positions = dspy.InputField(desc="The IDs and positions of all of the balls currently on the table..")
    target_balls = dspy.InputField(desc="The IDs of the balls that you must pot to win.")
    message = dspy.InputField(desc="A message from a user, use this to inform your decision on what shots to suggest.")
    number_of_shots = dspy.InputField(desc="The number of shots to suggest.")

    shots = dspy.OutputField(desc="The shots to suggest to the user, with each shot on a new numbered line, with no other text.")

class MultiShotSignature(dspy.Signature):
    """You are tasked with suggesting shots to make in a game of pool/billiards. Based on the message you recieve and the current state of the pool table, you must suggest shots to make that satisfy the users goal. 

    The IDs of the pockets are: left top (lt) at (0,2), right top (rt) at (1,2), left center (lc) at (0,1), right center (rc) at (1,1), left bottom (lb) at (0,0), right bottom (rb) at (1,0).
    
    The description of a shot is created using the events that you wish to occur, using the notation: "BALL-BALL-X-Y" for a collision between two balls X and Y, "BALL-POCKET-X-Z" for a ball X falling into pocket Z, and "BALL-CUSHION-X" for a ball X colliding with a cushion (note there is no second argument needed). Output the events of each shot as a comma separated list, with each shot on a new numbered line, for example:
    1. BALL-BALL-cue-blue, BALL-POCKET-blue-lb
    2. BALL-BALL-cue-red, BALL-CUSHION-red, BALL-POCKET-red-rc 
    3. ... 
    N. BALL-CUSHION-cue, BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt

    Try to be creative in your choice of events, but attempt to choose a shot that is simple and easy to execute. Make sure you do not repeat shots. Make sure a suggested shot does not foul by potting a ball that is not a target ball, hitting a non-target ball first, or by potting the cue ball.
    
    Use the Reasoning field to briefly think of what makes a good pool shot, and why you are choosing the shot you choose.
    """


    # --- START EXAMPLE ---

    # Ball positions:
    #     Ball cue: (0.69, 0.29)
    #     Ball yellow: (0.42, 0.78)
    #     Ball black: (0.75, 1.50)
    #     Ball pink: (0.43, 1.54)

    # Target Balls: red, blue, yellow

    # Message: Suggest 3 shots to make in this position.

    # Number of shots: 3

    # Reasoning:
    #     The only available target ball is the yellow ball. We must plan three shots that attempt to pot this ball in different ways. The yellow ball is near the left center lc pocket, and the cue ball is positioned so that the yellow ball is almost between the cue ball and the pocket. One shot would be to simply knock the yellow ball into the lc pocket. Another shot could attempt the further pocket, left top lt, as the cue ball and is also well alligned for this pocket. The black ball is between the yellow ball and the right top rt pocket so that is not a good option. Lastly, we could try to bounce the cue ball off the right cushion and into the yellow ball, potting it in the lc pocket.

    # Shots:
    #     1. BALL-BALL-cue-yellow, BALL-POCKET-yellow-lc
    #     2. BALL-BALL-cue-yellow, BALL-POCKET-yellow-lt
    #     3. BALL-CUSHION-cue, BALL-BALL-cue-yellow, BALL-POCKET-yellow-lc

    # --- END EXAMPLE ---

    ball_positions = dspy.InputField(desc="The IDs and positions of all of the balls currently on the table..")
    target_balls = dspy.InputField(desc="The IDs of the balls that you must pot to win.")
    message = dspy.InputField(desc="A message from a user, use this to inform your decision on what shots to suggest.")
    number_of_shots = dspy.InputField(desc="The number of shots to suggest.")

    shots = dspy.OutputField(desc="The shots to suggest to the user, with each shot on a new numbered line, with no other text.")


class LM_Suggester():

    def __init__(self, target_balls, vision=False):
        self.target_balls_str = ", ".join(target_balls)
        self.suggest_shots = dspy.ChainOfThought(MultiShotSignature) if not vision else dspy.ChainOfThought(MultiShotSignatureImage)
        self.record = {}
        self.vision = vision

    def __call__(self, env, state, message, N, lm, logger=None):
        if logger:
            logger.info(f"Suggesting {N} shots based on user message: {message}")
        
        if isinstance(state, dict):
            state = State.from_json(state)
            if logger:
                logger.info("Converted state from JSON to State object")

        current_balls_str = ""
        for ball, v in state.ball_positions.items():
            if not isinstance(v[0], str):
                current_balls_str += f"Ball {ball}: ({v[0]:.2f}, {v[1]:.2f})\n"
        if logger:
            logger.info(f"Current ball positions:\n{current_balls_str}")

        env.from_state(state)

        args_config = {
            "stop": ['Target Balls:', '\n\n']
        }
        if self.vision:
            args_config['image'] = env.get_image().image_base64

        if logger:
            logger.info(f"Calling suggest_shots with target_balls: {self.target_balls_str}")
        dspy.settings.configure(lm=lm)
        response = self.suggest_shots(
            ball_positions=current_balls_str,
            target_balls=self.target_balls_str,
            message=message,
            number_of_shots=str(N),
            config=args_config,
            lm=lm,
            
        )

        if logger:
            logger.info(f"Response from suggest_shots: {response}")

        self.record = {
            "current_balls": current_balls_str,
            "target_balls": self.target_balls_str,
            "message": message,
            "number_of_shots": N,
            "response": {
                "reasoning": response.rationale,
                "shots": response.shots
            }
        }

        if "I can't assist with that" in response.shots:
            if logger:
                logger.error(f"Error in shot suggestion: {response.shots}")
            return []
        
        shots_string = response.shots
        shots_events = []
        for idx in range(1, N+1):
            try:
                # Find the start of this shot's description
                start_index = shots_string.find(f"{idx}. ")
                if start_index == -1:
                    if logger:
                        logger.warning(f"Couldn't find shot {idx}.")
                    continue
                
                # Find the end of this shot's description (next newline or end of string)
                end_index = shots_string.find('\n', start_index)
                if end_index == -1:
                    end_index = len(shots_string)
                
                # Extract the shot description
                shot_description = shots_string[start_index + len(f"{idx}. "):end_index].strip()
                
                # Split the shot description and convert to Events
                shot = shot_description.split(", ")
                shot = [Event.from_encoding(s) for s in shot]

                shots_events.append(shot)
                if logger:
                    logger.info(f"Processed shot {idx}: {shot_description}")
            except Exception as e:
                logger.error(f"Error in processing shot {idx}: {e}")
                continue
        if logger:
            logger.info(f"Suggested {len(shots_events)} shots")
        return shots_events
