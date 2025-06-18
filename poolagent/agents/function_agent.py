import numpy as np
from typing import List, Tuple, Dict, Any
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from poolagent.pool import LIMITS, Fouls
from poolagent.utils import Agent
from .function_chooser import FunctionChooser

class FunctionAgent(Agent):
    def __init__(self, target_balls):
        super().__init__(target_balls, None)
        self.target_balls = target_balls
        self.chooser = FunctionChooser(target_balls)
        
        # Optimization parameters
        self.N = 5  # Number of optimization attempts
        self.ALPHA = 1e-2  # GP regression alpha
        self.INITIAL_SEARCH = 100  # Initial random points
        self.OPT_SEARCH = 20  # Optimization iterations
        
        self.record = {}

    def _create_optimizer(self, seed=None):
        """Create a new Bayesian Optimization object."""
        if seed is None:
            seed = np.random.randint(1e6)
            
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=lambda V0, phi, theta, a, b: 0,  # Placeholder function, as target_func is set later
            pbounds=LIMITS,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer

    def _calculate_reward(self, shot_params, env, initial_state, weight_distribution=None):
        """Calculate reward for a given shot, including penalties for extreme values."""
        V0, phi, theta, a, b = shot_params.values()
        shot = {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b}
        
        # Apply shot and check for fouls
        env.from_state(initial_state)
        foul = env.strike(**shot, check_rules=True, target_balls=self.target_balls)
        if foul != Fouls.NONE:
            return -1
        
        # Get state and events after shot
        state = env.get_state()
        state.params = shot
        events = env.get_events()
        
        # Calculate expected value
        _, _, expected_values, _, _ = self.chooser.evaluate_shots(
            initial_state, [shot], [events], [state], [foul], 
            weight_distribution=weight_distribution
        )
        
        # Apply penalties for extreme values
        penalty = (
            0.1 * (V0 / 4.0)**2 +
            0.1 * (abs(theta - 15) / 75)**2 +
            0.1 * (abs(a) / 0.25)**2 +
            0.1 * (abs(b) / 0.25)**2
        )
        
        return expected_values[0] - penalty

    def _optimize_single_shot(self, args: Tuple[Any, Any, Any, int]):
        """Perform a single optimization attempt."""
        env, initial_state, weight_distribution, seed = args
        
        # Create optimizer for this attempt
        optimizer = self._create_optimizer(seed)
        
        # Update the objective function
        optimizer.space.target_func = lambda V0, phi, theta, a, b: self._calculate_reward(
            {'V0': V0, 'phi': phi, 'theta': theta, 'a': a, 'b': b},
            env, initial_state, weight_distribution
        )
        
        # Perform optimization
        optimizer.maximize(
            init_points=self.INITIAL_SEARCH,
            n_iter=self.OPT_SEARCH
        )
        
        best_result = sorted(optimizer.res, key=lambda x: x['target'], reverse=True)[0]
        shot = best_result['params']
        
        # Simulate the shot to get state and events
        env.from_state(initial_state)
        foul = env.strike(**shot, check_rules=True, target_balls=self.target_balls)
        new_state = env.get_state()
        new_state.params = shot
        
        return shot, new_state, env.get_events(), foul

    def _process_shots_sequential(self, env, state, weight_distribution=None) -> Tuple[List, List, List, List]:
        """Process shots sequentially."""
        shots, states, events, fouls = [], [], [], []
        
        for i in range(self.N):
            shot, new_state, shot_events, foul = self._optimize_single_shot(
                (env, state, weight_distribution, np.random.randint(1e6))
            )
            shots.append(shot)
            states.append(new_state)
            events.append(shot_events)
            fouls.append(foul)
            
            if self.logger:
                self.logger.info(f"Completed optimization {i + 1}/{self.N}")
        
        return shots, states, events, fouls

    # def _process_shots_parallel(self, env, state, weight_distribution=None) -> Tuple[List, List, List, List]:
    #     """Process shots in parallel using multiprocessing."""
    #     args = [
    #         (env, state, weight_distribution, np.random.randint(1e6))
    #         for _ in range(self.N)
    #     ]
        
    #     # Use context manager with explicit process count
    #     with mp.Pool(processes=min(self.N, mp.cpu_count())) as pool:
    #         results = pool.map(self._optimize_single_shot, args)
            
    #     shots = [r[0] for r in results]
    #     states = [r[1] for r in results]
    #     events = [r[2] for r in results]
    #     fouls = [r[3] for r in results]
        
    #     return shots, states, events, fouls

    def take_shot(self, env, state, weight_distribution=None, logger=None, parallel=False, **kwargs) -> dict:
        """
        Take a shot using either sequential or parallel processing.
        
        Args:
            env: The pool environment
            state: Current state of the game
            weight_distribution: Optional weight distribution for shot evaluation
            logger: Optional logger for debugging
            parallel: Whether to use parallel processing (default: True)
        """
        self.logger = logger

        shots, states, events, fouls = self._process_shots_sequential(
            env, state, weight_distribution
        )
        
        # Choose the best shot
        best_shot_index, model_distributions, expected_values, raw_values, raw_difficulties = \
            self.chooser.evaluate_shots(state, shots, events, states, fouls)
        
        if logger:
            logger.info(f"Best shot: {shots[best_shot_index]}")
            logger.info(f"Expected value: {expected_values[best_shot_index]}")
        
        # Record results
        self.record = {
            'agent': 'FunctionAgent',
            'start_state': state.to_json(),
            'end_state': states[best_shot_index].to_json(),
            'shot': shots[best_shot_index],
            'shots': shots,
            'events': [e.to_json() for e in events[best_shot_index]],
            'best_shot_index': best_shot_index,
            'values': list(raw_values.tolist()),
            'difficulties': list(raw_difficulties.tolist()),
            'model_distributions': list(model_distributions.tolist()),
        }
        
        return shots[best_shot_index]

    @staticmethod
    def default_dict():
        return {"N": 5}