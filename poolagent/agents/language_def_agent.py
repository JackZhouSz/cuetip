import random
import time
from typing import List, Tuple, Dict, Any

from poolagent.utils import Agent, EventType
from .lm_suggester import LM_Suggester
from .lm_chooser import LM_Chooser
from .function_chooser import FunctionChooser
from poolagent.pool import Fouls
from poolagent.pool_solver import PoolSolver, Optimisers

class LanguageDEFAgent(Agent):
    def __init__(self, target_balls):
        super().__init__(target_balls, None)
        self.target_balls = target_balls
        self.solver = PoolSolver()
        self.suggester = LM_Suggester(target_balls)
        self.chooser = LM_Chooser(target_balls, defs=True)
        self.function_chooser = FunctionChooser(target_balls)
        self.SEARCH_STEPS = 100
        self.N = 5
        self.record = {}
        self.agent_name = "LanguageDEFAgent"

    def _simulate_shot(self, args: Tuple[int, List, Any, Any, List[int], int]) -> Tuple[Dict, Any, List, float, float, Any]:
        """Simulate a single shot with the given parameters."""
        idx, events, env, state, target_balls, search_steps = args
        env.reset()
        env.from_state(state)
        Optimisers.SEARCH_STEPS = search_steps
        t0 = time.time()
        
        result = self.solver.get_shot(env, state, events, target_balls)
        
        if self.logger:
            self.logger.info(f"Simulated shot {idx + 1}: {result[2]} in {time.time() - t0:.2f}s with rating {result[3]:.2f}")
            
        return result

    def _process_shots_sequential(self, env, state, intended_shots) -> Tuple[List, List, List]:
        """Process shots sequentially."""
        simulated_shot_params = []
        simulated_shot_events = []
        simulated_states = []
        
        for idx, events in enumerate(intended_shots):
            if self.logger:
                self.logger.info(f"Simulating shot {idx + 1}/{len(intended_shots)}: {events}")
            
            params, new_state, new_events, rating, std_dev, foul = self._simulate_shot(
                (idx, events, env, state, self.target_balls, self.SEARCH_STEPS)
            )
            
            if foul == Fouls.NONE:
                simulated_shot_params.append(params)
                simulated_shot_events.append(new_events)
                simulated_states.append(new_state)

        return simulated_shot_params, simulated_shot_events, simulated_states

    def take_shot(self, env, state, lm, message=None, logger=None, parallel=True) -> dict:
        """
        Take a shot using either sequential or parallel processing.
        
        Args:
            env: The pool environment
            state: Current state of the game
            lm: Language model for shot suggestion and choice
            message: Optional message for shot suggestion
            logger: Optional logger for debugging
            parallel: Whether to use parallel processing (default: True)
        """
        self.logger = logger  # Store logger for use in other methods
        
        # Get shot suggestions from LM
        self.intended_shot_events = self.suggester(
            env=env,
            state=state,
            message="Suggest the best shots to make in this position." if message is None else message,
            N=self.N,
            lm=lm,
            logger=logger
        )

        # Filter out null events, may be causing optimizer to fail
        for idx, shot_events in enumerate(self.intended_shot_events):
            events = []
            for event in shot_events:
                if len(event.encoding) == 0 or event.event_type == EventType.NULL:
                    # Null event - skip
                    continue
                events.append(event)
            self.intended_shot_events[idx] = events

        if logger:
            logger.info(f"Received intended shots: {self.intended_shot_events}")

        if not self.intended_shot_events:
            print("No shots suggested.")
            return env.random_params()

        # Process shots using either sequential or parallel approach
        if parallel:
            simulated_shot_params, simulated_shot_events, simulated_states = self._process_shots_parallel(
                env, state, self.intended_shot_events
            )
        else:
            simulated_shot_params, simulated_shot_events, simulated_states = self._process_shots_sequential(
                env, state, self.intended_shot_events
            )

        if not simulated_shot_params:
            print("No valid shots found.")
            return env.random_params()
        
        def_values = {}
        # Choose the best shot
        _ = self.function_chooser(
            starting_state=state,
            final_states=simulated_states,
            shot_params=simulated_shot_params,
            shot_events=simulated_shot_events
        )
        values = self.function_chooser.record['values']
        difficulties = self.function_chooser.record['difficulties']

        value_str = ""
        for idx, shot_values in enumerate(values):
            shot_values_str = ", ".join([f"{v:.2f}" for v in shot_values])
            value_str += f"{idx}: {shot_values_str}\n"

        difficulty_str = ""
        for idx, shot_difficulties in enumerate(difficulties):
            shot_difficulties_str = ", ".join([f"{v:.2f}" for v in shot_difficulties])
            difficulty_str += f"{idx}: {shot_difficulties_str}\n"

        def_values['value_rules'] = value_str
        def_values['difficulty_rules'] = difficulty_str

        # Choose the best shot
        try:
            chosen_shot = self.chooser(
                shot_events=simulated_shot_events, 
                lm=lm, 
                logger=logger,
                def_values=def_values
            )
        except:
            print(f"Error in {self.agent_name} - failed to choose shot")
            chosen_shot = random.randint(0, len(simulated_shot_params) - 1)

        if logger:
            logger.info(f"Chosen shot: {chosen_shot}")

        assert 0 <= chosen_shot < len(simulated_shot_events), \
            f"Chosen shot must be between 0 and {len(simulated_shot_events) - 1}, got {chosen_shot}."

        # Record the shot details
        self.record = {
            'agent': self.agent_name,
            'start_state': state.to_json(),
            'end_state': simulated_states[chosen_shot].to_json(),
            'shot': simulated_shot_params[chosen_shot],
            'shots': simulated_shot_params,
            'events': [e.to_json() for e in simulated_shot_events[chosen_shot]],
            'lm_chooser': self.chooser.record,
            'lm_suggester': self.suggester.record,
            'function_chooser': self.function_chooser.record,
        }

        return simulated_shot_params[chosen_shot]

    @staticmethod
    def default_dict():
        return {"N": 10}

class VisionLanguageDEFAgent(LanguageDEFAgent):
    def __init__(self, target_balls):
        super().__init__(target_balls)
        self.suggester = LM_Suggester(target_balls, vision=True)
        self.agent_name = "VisionLanguageDEFAgent"

    @staticmethod
    def default_dict():
        return {"N": 10}