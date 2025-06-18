
import os

from typing import List, Dict, Tuple
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer, UtilityFunction
from scipy import optimize

from poolagent.pool import Pool, Fouls
from poolagent.utils import *
from poolagent.pool import LIMITS

class Optimisers:

    _BayesOptimiser = 0
    _SimulatedAnnealing = 1
    _BruteForce = 2

    default_params = {
        "V0": 1,
        "theta": 14,
        "phi": 0,
        "a": 0,
        "b": 0.5
    }

    ### Search Parameters
    ###    - Initial random points to sample in the parameter space
    ###    - Bayesian opt search steps to perform
    INITIAL_RANDOM = 250
    SEARCH_STEPS = 100

    ### Below this threshold, we consider a ball to be sufficiently close to a target position
    DISTANCE_THRESHOLD = 0.25

    VALID_DESCRIPTIONS = {
        "any":           (0.5, 1.0),
        "top" :          (0.5, 1.5),
        "bottom" :       (0.5, 0.5),
        "top left" :     (0.25, 1.75),
        "top right" :    (0.75, 1.75),
        "bottom left" :  (0.25, 0.25),
        "bottom right" : (0.75, 0.25),
    }

    BAYES_OPTIMIZER = None
    BOUNDS_TRANSFORMER = None

    @staticmethod
    def BayesOptimiser(reward_function, param_space : dict = LIMITS) -> Tuple[dict, float, float]:

        ### Bounds Transformer focuses the search on a smaller region of the parameter space as the search progresses, good and bad
        Optimisers.BOUNDS_TRANSFORMER = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0,
        )
        Optimisers.BAYES_OPTIMIZER = BayesianOptimization(
            f=reward_function,
            pbounds=param_space,
            random_state=1,
            bounds_transformer=Optimisers.BOUNDS_TRANSFORMER
        )

        ### Initial random points to sample in the parameter space
        search_done = False
        for i in range(Optimisers.INITIAL_RANDOM):
            next_point = {
                "V0": np.random.uniform(*param_space["V0"]),
                "phi": np.random.uniform(*param_space["phi"]),
                "theta": np.random.uniform(*param_space["theta"]),
                "a": np.random.uniform(*param_space["a"]),
                "b": np.random.uniform(*param_space["b"])
            }
            target = reward_function(**next_point)
            Optimisers.BAYES_OPTIMIZER.register(params=next_point, target=target)
            print(f"Initial random point: {i}/{Optimisers.INITIAL_RANDOM} with reward: {target}")

            if float(Optimisers.BAYES_OPTIMIZER.max["target"])>0.95:
                search_done = True
                break

        ### Bayesian Optimisation
        if not search_done:
            utility = UtilityFunction(kind="ucb", kappa=2.5, xi=0.0)
            for i in range(Optimisers.SEARCH_STEPS):
                next_point = Optimisers.BAYES_OPTIMIZER.suggest(utility)
                target = reward_function(**next_point)
                Optimisers.BAYES_OPTIMIZER.register(params=next_point, target=target)
                print(f"Search point: {i}/{Optimisers.SEARCH_STEPS} with reward: {target}")

                if float(Optimisers.BAYES_OPTIMIZER.max["target"])>0.95:
                    break

        best_shot = Optimisers.BAYES_OPTIMIZER.max
        
        # Get variance of the best shot
        _, std = Optimisers.BAYES_OPTIMIZER._gp.predict(np.array(
            [
                best_shot["params"]["V0"], 
                best_shot["params"]["phi"],
                best_shot["params"]["theta"],
                best_shot["params"]["a"],
                best_shot["params"]["b"]    
            ]
        ).reshape(1, -1), return_std=True)

        # Convert std array to float 
        std = float(sum(std) / len(std))
        
        return {
            "V0": float(best_shot["params"]["V0"]),
            "phi": float(best_shot["params"]["phi"]),
            "theta": float(best_shot["params"]["theta"]),
            "a": float(best_shot["params"]["a"]),
            "b": float(best_shot["params"]["b"])
        }, float(best_shot["target"]), std

    @staticmethod
    def BruteForce(reward_function, param_space : dict = LIMITS, Ns : int = 2) -> Tuple[dict, float, float]:

        def f(x):
            return 1.0 - reward_function(x[0], x[1], x[2], x[3], x[4])
        
        OPTIONS = {
            "Ns": Ns
        }

        x0 = optimize.brute(f, ranges = (
            (param_space["V0"][0], param_space["V0"][1]),
            (param_space["phi"][0], param_space["phi"][1]),
            (param_space["theta"][0], param_space["theta"][1]),
            (param_space["a"][0], param_space["a"][1]),
            (param_space["b"][0], param_space["b"][1])
        ), **OPTIONS)
        fval = f(x0)

        return {
            "V0": x0[0],
            "phi": x0[1],
            "theta": x0[2],
            "a": x0[3],
            "b": x0[4]
        }, 1.0 - fval, 0

    @staticmethod
    def SimulatedAnnealing(reward_function, param_space : dict = LIMITS, x0: dict = None, verbose : bool=True) -> Tuple[dict, float, float]:

        def f(x):
            return 1.0 - reward_function(x[0], x[1], x[2], x[3], x[4])
        
        if x0:
            x0 = np.array([x0["V0"], x0["phi"], x0["theta"], x0["a"], x0["b"]])

        all_results = []
        def callback(x, f, context):

            shot = x.copy()
            shot = {
                "V0": shot[0],
                "phi": shot[1],
                "theta": shot[2],
                "a": shot[3],
                "b": shot[4]
            }
            r = 1.0 - f

            if verbose:
                x_str = f"V0: {x[0]:.2f}, phi: {x[1]:.2f}, theta: {x[2]:.2f}, a: {x[3]:.2f}, b: {x[4]:.2f}"
                print("-"*50)
                print(f"x ------> {x_str}")
                print(f"f(x) ---> {r:.3f}")

            all_results.append((shot, r))

        OPTIONS = {
            "bounds": [
                param_space["V0"], 
                param_space["phi"], 
                param_space["theta"], 
                param_space["a"], 
                param_space["b"]
            ],
            "maxiter": Optimisers.SEARCH_STEPS,
            "no_local_search": True,
            "minimizer_kwargs": {
                "tol": 0.01,
            },
            "x0": x0,
            "callback": callback
        }

        res = optimize.dual_annealing(f, **OPTIONS)

        return {
            "V0": res.x[0],
            "phi": res.x[1],
            "theta": res.x[2],
            "a": res.x[3],
            "b": res.x[4]
        }, 1.0 - res.fun, all_results

    @staticmethod
    def description_reward(description : str, events : List[Event]) -> float:
        """
        Reward function for the description of the end state
        """

        description = description.replace(".", "")
        description = description.lower().strip()
        
        # Check if the description is valid
        if not description in Optimisers.VALID_DESCRIPTIONS.keys():
            return 0
        
        # If the description is any, then we return 1
        if description == "any":
            return 1

        # Check if the cue ball stop event is in the events
        cue_stop_event = None
        for e in events:
            if e.event_type == EventType.BALL_STOP and e.arguments[0] == "cue" and e.pos:
                cue_stop_event = e
                break
        if not cue_stop_event:
            # Likely potted the cue ball
            return 0
        
        target_position = Optimisers.VALID_DESCRIPTIONS[description]  
        cue_stop_event_position = cue_stop_event.pos

        euclidean_distance = np.linalg.norm(
            np.array(cue_stop_event_position) - np.array(target_position)
        )

        # If the distance is less than a threshold, then we return 1, as we are close enough and any further optimization is not needed
        if euclidean_distance < Optimisers.DISTANCE_THRESHOLD:
            return 1.0
        
        return 1.0 - np.clip(euclidean_distance / 2.0, 0, 1)

    @staticmethod
    def event_reward(events : List[Event], new_events : List[Event]) -> float:
        """
        Reward function for the event sequence
        """

        if Event.ball_pocket("cue") in new_events:
            return 0

        ord_r = 0
        for idx, event in enumerate(events):
            
            if event in new_events:
                ord_r += (idx + 1) 
                new_events = new_events[new_events.index(event)+1:]

            else:
                # Check for near misses
                closest_event = Event.get_closest_event(event, new_events)
                if closest_event:
                    ord_r += Event.distance(event, closest_event)

                # If there is no near miss, then we break as order must be preserved
                break

                
        return ord_r / sum([i+1 for i in range(len(events))])
    
    @staticmethod
    def length_reward(intended_events : List[Event], actual_events : List[Event]) -> float:
        """
        Reward function for the length of the event sequence, reward approaches 0 as length of new_events increases
        """

        r_len = len(actual_events) - ( 2 + len(intended_events) ) # plus two for hitting cue ball and cue ball stopping
        return 0.9 ** r_len
    
    @staticmethod
    def param_magnitude_reward(params : Dict[str, float]) -> float:
        """
        Reward function for the magnitude of the parameters
        """
        V0 = params["V0"]
        phi = params["phi"]
        theta = params["theta"]
        a = params["a"]
        b = params["b"]

        penalty = (
            0.1 * (V0 / 4.0)**2 +
            0.1 * (abs(theta - 15) / 75)**2 +
            0.1 * (abs(a) / 0.25)**2 +
            0.1 * (abs(b) / 0.25)**2
        )

        return 1.0 - penalty

class PoolSolver:
    def __init__(self, solver_type=Optimisers._SimulatedAnnealing):
        self.verbosity = os.getenv("VERBOSITY", "NONE")

        self.optimiser = Optimisers.BayesOptimiser
        if solver_type == Optimisers._SimulatedAnnealing:
            self.optimiser = Optimisers.SimulatedAnnealing
        elif solver_type == Optimisers._BruteForce:
            self.optimiser = Optimisers.BruteForce
    

    # TODO: make the return value a struct of some kind 
    def get_shot(self, env : Pool, state : State, events : List[Event], target_balls : List[str]) -> Tuple[Dict[str, float], State, List[Event], float, float, bool]:

        # 1. embed end state text
        # 2. perform search in param space 
        # 3. optimise for the params that cause the event sequence and end state is close to the embedded end state text

        # Some simple checks about the board
        if len(events) == 0:
            return Optimisers.default_params, state, [], 0, 1, False
        elif state.is_potted("cue"):
            return Optimisers.default_params, state, [], 0, 1, False
        elif all([state.is_potted(ball) for ball in target_balls]):
            return Optimisers.default_params, state, [], 1, 1, True

        INITIAL_STATE = state
        POOL = env

        def reward_function(V0, phi, theta, a, b) -> float:
            
            params = {
                "V0": V0,
                "theta": theta,
                "phi": phi,
                "a": a,
                "b": b
            }

            return PoolSolver.rate_shot(POOL, INITIAL_STATE, events, params, target_balls)
        
        params, rating, std_dev = self.optimiser(reward_function)

        POOL.from_state(INITIAL_STATE)
        foul = POOL.strike(**params, check_rules=True, target_balls=target_balls)

        if foul != Fouls.NONE and self.verbosity == "INFO":
            print(f"SHOT FOULED: {Fouls.string(foul)}")

        new_events = POOL.get_events()

        if self.verbosity == "INFO":
            print(f"Found shot: {params} with rating: {rating}")

        new_state = POOL.get_state()
        new_state.params = params.copy()

        return params, new_state, new_events, rating, std_dev, foul
        
    @staticmethod
    def rate_shot(env : Pool, state : State, events : List[Event], params : Dict[str, float], target_balls : List[str]) -> float:
        env.from_state(state)
        try:
            foul = env.strike(**params, check_rules=True, target_balls=target_balls)
        except Exception as e:
            return 0
        
        if foul != Fouls.NONE:
            return 0

        new_events = env.get_events()
        
        rewards = {
            "event": (0.7, Optimisers.event_reward(events, new_events)),
            "length": (0.2, Optimisers.length_reward(events, new_events)),
            "param_mag": (0.1, Optimisers.param_magnitude_reward(params)),
        }

        return sum([v[0] * v[1] for v in rewards.values()])
        
