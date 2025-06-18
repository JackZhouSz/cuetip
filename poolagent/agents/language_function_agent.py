import time
from functools import partial
from typing import List, Tuple, Dict, Any
from poolagent.utils import Agent, EventType
from .lm_suggester import LM_Suggester
from .function_chooser import FunctionChooser
from poolagent.pool import Fouls
from poolagent.pool_solver import PoolSolver, Optimisers

class LanguageFunctionAgent(Agent):
    def __init__(self, target_balls):
        super().__init__(target_balls)
        self.target_balls = target_balls
        self.solver = PoolSolver()
        self.suggester = LM_Suggester(target_balls)
        self.chooser = FunctionChooser(target_balls)
        self.N = 5
        self.SEARCH_STEPS = 200
        self.record = {}

    def _simulate_shot(self, args: Tuple[List, Any, Any, List[int], int]) -> Tuple[Dict, Any, List]:
        """Simulate a single shot with the given parameters."""
        events, env, state, target_balls, search_steps = args
        Optimisers.SEARCH_STEPS = search_steps
        t0 = time.time()
        
        params, new_state, new_events, rating, std_dev, foul = self.solver.get_shot(
            env, state, events, target_balls
        )
        
        if self.logger:
            self.logger.info(f"Simulated shot: {new_events} in {time.time() - t0:.2f}s with rating {rating:.2f}")
        
        if foul == Fouls.NONE:
            return (params, new_state, new_events)
        return None

    def _process_shots_sequential(self, env, state, intended_shots) -> Tuple[List, List, List]:
        """Process shots sequentially."""
        results = []
        for idx, events in enumerate(intended_shots):
            if self.logger:
                self.logger.info(f"Simulating shot {idx + 1}/{len(intended_shots)}: {events}")
            
            result = self._simulate_shot(
                (events, env, state, self.target_balls, self.SEARCH_STEPS)
            )
            if result is not None:
                results.append(result)

        # Unpack results
        shot_params, final_states, shot_events = [], [], []
        for params, new_state, new_events in results:
            shot_params.append(params)
            final_states.append(new_state)
            shot_events.append(new_events)

        return shot_params, final_states, shot_events

    # def _process_shots_parallel(self, env, state, intended_shots) -> Tuple[List, List, List]:
    #     """Process shots in parallel using multiprocessing."""
    #     # Prepare arguments for parallel processing
    #     args = [
    #         (events, env, state, self.target_balls, self.SEARCH_STEPS)
    #         for events in intended_shots
    #     ]
        
    #     # Use multiprocessing to simulate shots in parallel
    #     with mp.Pool(processes=min(self.N, mp.cpu_count())) as pool:
    #         results = pool.map(self._simulate_shot, args)
            
    #     # Filter out None results and unpack
    #     shot_params, final_states, shot_events = [], [], []
    #     for result in results:
    #         if result is not None:
    #             params, new_state, new_events = result
    #             shot_params.append(params)
    #             final_states.append(new_state)
    #             shot_events.append(new_events)

    #     return shot_params, final_states, shot_events

    def take_shot(self, env, state, lm, message=None, logger=None, parallel=False) -> dict:
        """
        Take a shot using either sequential or parallel processing.
        
        Args:
            env: The pool environment
            state: Current state of the game
            lm: Language model for shot suggestion
            logger: Optional logger for debugging
            parallel: Whether to use parallel processing (default: False)
        """
        self.logger = logger

        # Get shot suggestions from LM
        intended_shots = self.suggester(
            env=env,
            state=state,
            message=f"Suggest the {self.N} best shots to make in this position." if message is None else message,
            N=self.N,
            lm=lm,
            logger=logger
        )

        # Filter out null events, may be causing optimizer to fail
        for idx, shot_events in enumerate(intended_shots):
            events = []
            for event in shot_events:
                if len(event.encoding) == 0 or event.event_type == EventType.NULL:
                    # Null event - skip
                    continue
                events.append(event)
            intended_shots[idx] = events

        if logger:
            logger.info(f"Intended shot events: {intended_shots}")

        if not intended_shots:
            print("No shots suggested.")
            return env.random_params()

        # Process shots using either sequential or parallel approach
        if parallel:
            shot_params, final_states, shot_events = self._process_shots_parallel(
                env, state, intended_shots
            )
        else:
            shot_params, final_states, shot_events = self._process_shots_sequential(
                env, state, intended_shots
            )

        if not shot_params:
            print("No valid shots found.")
            return env.random_params()

        # Choose the best shot
        best_shot_index = self.chooser(
            starting_state=state,
            final_states=final_states,
            shot_params=shot_params,
            shot_events=shot_events
        )

        if logger:
            logger.info(f"Chosen shot: {best_shot_index}")

        assert 0 <= best_shot_index < len(shot_events), \
            f"Chosen shot must be between 0 and {len(shot_events) - 1}, got {best_shot_index}."

        # Record the shot details
        self.record = {
            'agent': 'LanguageFunctionAgent',
            'best_shot_index': best_shot_index,
            'start_state': state.to_json(),
            'end_state': final_states[best_shot_index].to_json(),
            'shot': shot_params[best_shot_index],
            'shots': shot_params,
            'events': [e.to_json() for e in shot_events[best_shot_index]],
            'lm_suggester': self.suggester.record,
            'function_chooser': self.chooser.record
        }

        return shot_params[best_shot_index]

    @staticmethod
    def default_dict():
        return {
            "N": 10,
            "model": "gpt-4o",
            "backend": "azure"
        }

class VisionLanguageFunctionAgent(LanguageFunctionAgent):
    def __init__(self, target_balls):
        super().__init__(target_balls)
        self.suggester = LM_Suggester(target_balls, vision=True)

    @staticmethod
    def default_dict():
        return {
            "N": 10,
            "model": "gpt-4o",
            "backend": "azure"
        }