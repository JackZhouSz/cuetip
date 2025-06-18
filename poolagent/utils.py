import base64, io, PIL, time, json, os, typing, types, random, sys, concurrent, itertools, dsp, dspy
import numpy as np
from PIL import ImageDraw

from typing import Tuple, List, Dict
from io import StringIO
from math import ceil
from collections import OrderedDict

SHOT_PARAM_VALUES = {
    "V0": {"mean": 1.5, "std": 1, "min": 0.25, "max": 3.5},
    "theta": {"mean": 20, "std": 7.5, "min": 5, "max": 60},
    "phi": {"mean": 180, "std": 60, "min": 0, "max": 360},
    "a": {"mean": 0, "std": 0.1, "min": -0.25, "max": 0.25},
    "b": {"mean": 0, "std": 0.1, "min": -0.25, "max": 0.25}
}

def dspy_setup(in_config):

    if not 'backend' in in_config:
        backend = "azure"
    else:
        backend = in_config['backend']


    if backend == "openai":

        keys = [
            "temperature",
            "max_tokens",
            "model",
        ]
        config = {k: v for k, v in in_config.items() if k in keys}

        openai_llm = dspy.OpenAI(
            **config
        )
        dspy.settings.configure(lm=openai_llm)
        print(f"Initialised OpenAI LLM - {config['model']}")
        return openai_llm

    elif backend == "ollama":

        keys = [
            "temperature",
            "max_tokens",
            "model",
            "num_ctx"
        ]
        config = {k: v for k, v in in_config.items() if k in keys}

        ollama_llm = dspy.OllamaLocal(
            model_type="text",
            **config
        )
        dspy.settings.configure(lm=ollama_llm)
        print(f"Initialised Ollama LLM - {config['model']}")
        return ollama_llm

    elif backend == "huggingface":
            
            keys = [
                "temperature",
                "max_tokens",
                "model",
            ]
            config = {k: v for k, v in in_config.items() if k in keys}
    
            huggingface_llm = dspy.HFModel(
                **config
            )
            dspy.settings.configure(lm=huggingface_llm)
            print(f"Initialised HuggingFace LLM - {config['model']}")
            return huggingface_llm

    elif backend == "azure":
        import dotenv 
        dotenv.load_dotenv()
        assert os.getenv("API_KEY") is not None, "API_KEY not found in .env file, this is required for the Azure OpenAI API"
        assert os.getenv("API_BASE") is not None, "API_BASE not found in .env file, this is required for the Azure OpenAI API"
        api_key = os.getenv("API_KEY")
        base_url = os.getenv("API_BASE")
        api_version = '2024-06-01'

        keys = [
            "temperature",
            "max_tokens",
            "model",
        ]
        config = {k: v for k, v in in_config.items() if k in keys}

        azure_llm = dspy.AzureOpenAI(
            api_base=base_url,
            api_version=api_version,
            api_key=api_key,
            **config
        )
        dspy.settings.configure(lm=azure_llm)
        print(f"Initialised Azure OpenAI LLM - {config['model']}")
        return azure_llm

    else:

        raise ValueError("Invalid backend specified, must be one of 'openai', 'ollama', 'huggingface', or 'azure'")

def plt_best_shot(values, difficulties, averages):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(20, 10))

    # Subtract averages
    values_adjusted = np.array(values)  - np.array(averages['value'])
    difficulties_adjusted = np.array(difficulties)  - np.array(averages['difficulty'])

    # Plot values
    colors_values = ['red' if v < 0 else 'blue' for v in values_adjusted]
    ax[0].bar(range(len(values_adjusted)), values_adjusted, color=colors_values)
    ax[0].set_title("Values")
    ax[0].set_xlabel("Function")
    ax[0].set_ylabel("Value")
    ax[0].set_xticks(range(len(values_adjusted)))
    ax[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    # Plot difficulties
    colors_difficulties = ['red' if d < 0 else 'blue' for d in difficulties_adjusted]
    ax[1].bar(range(len(difficulties_adjusted)), difficulties_adjusted, color=colors_difficulties)
    ax[1].set_title("Difficulties")
    ax[1].set_xlabel("Function")
    ax[1].set_ylabel("Difficulty")
    ax[1].set_xticks(range(len(difficulties_adjusted)))
    ax[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()

def print_distribution(distribution):
    distribution = np.array(distribution).squeeze()
    max_bar_width = 40  # Maximum width of the bar
    expected_value = sum([(i/10) * prob for i, prob in enumerate(distribution)])
    print("\nProbability Distribution:")
    print("-------------------------")
    for i, prob in enumerate(distribution):
        bar_width = int(prob * max_bar_width)
        bar = '█' * bar_width
        print(f"{i/10:.1f}: {bar} {prob:.2f}")
    print(f"\nExpected Value: {expected_value:.2f}")
    print("-------------------------")

def plot_heatmaps(best_index, values, difficulties, VISUALISATIONS_DIR="visualisations"):
    import matplotlib.pyplot as plt

    # Transpose the matrices
    values = values.T if isinstance(values, np.ndarray) else np.array(values).T
    difficulties = difficulties.T if isinstance(difficulties, np.ndarray) else np.array(difficulties).T
    
    N_values, S = values.shape
    N_difficulties = difficulties.shape[0]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot last_values heatmap
    im1 = ax1.imshow(values, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax1.set_title('Values')
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Plot last_difficulties heatmap
    im2 = ax2.imshow(difficulties, cmap='viridis', vmin=0, vmax=1, aspect='equal')
    ax2.set_title('Difficulties')
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Highlight the chosen column
    rect1 = plt.Rectangle((best_index-0.5, -0.5), 1, N_values, fill=False, edgecolor='red', linewidth=2)
    rect2 = plt.Rectangle((best_index-0.5, -0.5), 1, N_difficulties, fill=False, edgecolor='red', linewidth=2)
    ax1.add_patch(rect1)
    ax2.add_patch(rect2)
    
    # Set x and y axis labels
    ax1.set_xticks(np.arange(S))
    ax1.set_yticks(np.arange(N_values))
    ax1.set_xticklabels([f'S{i+1}' for i in range(S)])
    ax1.set_yticklabels([f'N{i+1}' for i in range(N_values)])
    ax1.set_xlabel('States')
    ax1.set_ylabel('Functions (Values)')
    
    ax2.set_xticks(np.arange(S))
    ax2.set_yticks(np.arange(N_difficulties))
    ax2.set_xticklabels([f'S{i+1}' for i in range(S)])
    ax2.set_yticklabels([f'N{i+1}' for i in range(N_difficulties)])
    ax2.set_xlabel('States')
    ax2.set_ylabel('Functions (Difficulties)')
    
    # Rotate the tick labels and set their alignment
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Adjust layout and display the plot
    plt.tight_layout()
    # plt.show()
    if not os.path.exists(VISUALISATIONS_DIR):
        os.makedirs(VISUALISATIONS_DIR)
    plt.savefig(f"{VISUALISATIONS_DIR}/tmp_heatmap.png")

class SKILL_LEVELS:

    NOVICE = {
        "V0": 1.25,
        "theta": 1.5,
        "phi": 1.25,
        "a": 0.2,
        "b": 0.2
    }

    AMATEUR = {
        "V0": 1,
        "theta": 1.25,
        "phi": 1.0,
        "a": 0.15,
        "b": 0.15
    }

    PRO = {
        "V0": 0.75,
        "theta": 1.0,
        "phi": 0.75,
        "a": 0.1,
        "b": 0.1
    }

    BASELINE = {
        "V0": 0.075,
        "theta": 0.1,
        "phi": 0.125,
        "a": 0.05,
        "b": 0.05
    }

    NONE = {
        "V0": 0,
        "theta": 0,
        "phi": 0,
        "a": 0,
        "b": 0
    }

def blur_shot(shot_params, skill=SKILL_LEVELS.AMATEUR):
    shot = {
        "V0": np.random.normal(shot_params["V0"], skill["V0"]),
        "theta":  np.random.normal(shot_params["theta"], skill["theta"]),
        "phi": np.random.normal(shot_params["phi"], skill["phi"]) % 360,
        "a": np.random.normal(shot_params["a"], skill["a"]),
        "b": np.random.normal(shot_params["b"], skill["b"])
    }
    for k in shot:
        shot[k] = np.clip(shot[k], SHOT_PARAM_VALUES[k]['min'], SHOT_PARAM_VALUES[k]['max'])

    return shot

class Agent():
    def __init__(self, target_balls, config=None, **kwargs):
        self.target_balls = target_balls
        self.config = config
        self.record = {}

    def take_shot(self, env, state) -> dict:
        pass

    @staticmethod
    def default_dict():
        return {}
    
def random_ball_shot(game, target_balls : List[str] = None, skill=SKILL_LEVELS.AMATEUR) -> dict:

    if not target_balls:
        target_balls = game.target_balls[game.current_player]

    state = game.get_state()
    target_balls = [ball for ball in target_balls if not state.is_potted(ball)]

    if not target_balls:
        return game.random_params()

    ball = random.choice(target_balls)
    phi = game.attempt_pot_ball(ball)

    if phi == -1:
        return game.random_params()
    
    if skill == SKILL_LEVELS.NONE:
        return {
            "V0": 2 ,
            "theta": 14,
            "phi": phi,
            "a": 0,
            "b": 0
        }

    return {
        "V0": 2 + random.uniform(-skill["V0"], skill["V0"]),
        "theta": 14 + random.uniform(-skill["theta"], skill["theta"]),
        "phi": abs(phi + random.uniform(-skill["phi"], skill["phi"])),
        "a": random.uniform(-skill["a"], skill["a"]),
        "b": random.uniform(-skill["b"], skill["b"])
    }

def start_pheonix():
    # Phoenix Visualization setup 
    import phoenix as px
    phoenix_session = px.launch_app()
    from openinference.instrumentation.dspy import DSPyInstrumentor
    from opentelemetry import trace as trace_api
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk import trace as trace_sdk
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    endpoint = "http://127.0.0.1:6006/v1/traces"
    resource = Resource(attributes={})
    tracer_provider = trace_sdk.TracerProvider(resource=resource)
    span_otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter=span_otlp_exporter))

    trace_api.set_tracer_provider(tracer_provider=tracer_provider)
    DSPyInstrumentor().instrument()

def validate_library_function(function_code) -> Tuple[bool, str]:
    """Attempt to validate a user-defined function by executing it in a restricted environment. If it executes successfully, the function is considered valid and a success message is returned. If an error occurs, the error message is returned.

    Args:
        function_code (_type_): The code defining the function to validate

    Raises:
        ValueError: If no function is defined in the provided code

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating whether the function is valid and a message describing the result (an error message if an error occurs).
    """
    # Set up a restricted environment
    restricted_globals = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'chr': chr, 'dict': dict,
            'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'format': format, 'frozenset': frozenset,
            'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
            'len': len, 'list': list, 'map': map, 'hasattr': hasattr,
            'max': max, 'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str, '__name__': __name__,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip, '__build_class__': __build_class__,
            'List': typing.List, 'Tuple': typing.Tuple, 'Dict': typing.Dict, 'State': State,
            'random': random, '__import__': __import__, 'itertools': itertools, 'next': next,
            'np': np
        }
    }

    try:
        # Compile the function code
        compiled_code = compile(function_code, '<string>', 'exec')
        
        # Execute the code to define the function
        exec(compiled_code, restricted_globals)
        
        # Extract the defined function
        user_function = next((v for v in restricted_globals.values() 
                            if isinstance(v, types.FunctionType)), None)
        
        if user_function is None:
            raise ValueError("No function was defined in the provided code")
        
        return True, ""
    except Exception as e:
        return False, f"Error: {str(e)}"

def safe_exec_function(function_code, args: List = []) -> Tuple[bool, str]:
    """Safely execute a user-defined function in a restricted environment, capturing any errors that occur during execution. The function is called with the provided state, states, and target_balls arguments.

    Args:
        function_code (_type_): The code defining the function to execute
        args (List, optional): The arguments to pass to the function. Defaults to [].

    Raises:
        ValueError: If no function is defined in the provided code

    Returns:
        _type_: A tuple containing the result of the function and any printed output (error messages or other output)
    """

    restricted_globals = {
        '__builtins__': {
            'abs': abs, 'all': all, 'any': any, 'ascii': ascii,
            'bin': bin, 'bool': bool, 'chr': chr, 'dict': dict,
            'divmod': divmod, 'enumerate': enumerate, 'filter': filter,
            'float': float, 'format': format, 'frozenset': frozenset,
            'hash': hash, 'hex': hex, 'int': int, 'isinstance': isinstance,
            'len': len, 'list': list, 'map': map, 'hasattr': hasattr,
            'max': max, 'min': min, 'oct': oct, 'ord': ord, 'pow': pow, 'print': print,
            'range': range, 'repr': repr, 'reversed': reversed, 'round': round,
            'set': set, 'slice': slice, 'sorted': sorted, 'str': str, '__name__': __name__,
            'sum': sum, 'tuple': tuple, 'type': type, 'zip': zip, '__build_class__': __build_class__,
            'List': typing.List, 'Tuple': typing.Tuple, 'Dict': typing.Dict, 'State': State,
            'random': random, '__import__': __import__, 'Event': Event, 'itertools': itertools, 'next': next,
            'np': np
        }
    }

    # Redirect stdout to capture print output
    old_stdout = sys.stdout
    redirected_output = sys.stdout = StringIO()

    try:
        # Compile the function code
        compiled_code = compile(function_code, '<string>', 'exec')
        
        # Execute the code to define the function
        exec(compiled_code, restricted_globals)
        
        # Extract the defined function
        user_function = next((v for v in restricted_globals.values() 
                                if isinstance(v, types.FunctionType)), None)
        
        if user_function is None:
            raise ValueError("No function was defined in the provided code")
        
        # Call the function with the state object
        result = user_function(*args)
        
        # Get any printed output
        output = redirected_output.getvalue()
        
        return result, output
    except Exception as e:
        return None, f"Error: {str(e)}"
    finally:
        # Reset stdout
        sys.stdout = old_stdout

def mse_loss(y_true, y_pred):
    """Compute mean squared error loss for the entire dataset in a vectorized manner.

    Args:
        y_true (numpy.ndarray): True labels. Shape: (n_samples)
        y_pred (numpy.ndarray): Predicted probabilities. Shape: (n_samples)
    """
    return np.mean((y_true - y_pred) ** 2)

def cross_entropy_loss(y_true, y_pred, epsilon=1e-15):
    """Compute cross-entropy loss for the entire dataset in a vectorized manner.
    
    Args:
        y_true (numpy.ndarray): True labels. Shape: (n_samples, n_classes)
        y_pred (numpy.ndarray): Predicted probabilities. Shape: (n_samples, n_classes)
        epsilon (float): Small constant to avoid log(0)
    
    Returns:
        float: Average cross-entropy loss
    """
    # Clip predictions to avoid log(0)
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    loss = -np.mean(y_true * np.log(y_pred), axis=1) 
    return np.mean(loss)

def eval_learned_function(learned_function: str, dataset, target_balls, evolve_type: str = 'value', threading_eval: bool = True, N_THREADS=8, TIMEOUT=5):
    """Evaluate a learned function on a dataset using either single or multiple threads.

    Args:
        learned_function (str): The learned function code to evaluate
        dataset (dict): The evaluation dataset, containing 'train' and 'test' subsets
        target_balls (list): The target balls (used only if evolve_type is 'value')
        evolve_type (str): The type of evolution being performed, either 'value' or 'difficulty'. Defaults to 'value'.
        threading_eval (bool): Whether to use threading for evaluation. Defaults to True.
        N_THREADS (int): Number of threads to evaluate on. Defaults to 8.
        TIMEOUT (int): The timeout value for a thread, in case it hangs. Defaults to 5.

    Returns:
        tuple: A tuple containing the losses, accuracies, and any output or errors that occurred during evaluation.
    """
    losses = {'train': 0, 'test': 0}
    accuracies = {'train': 0, 'test': 0}
    DEFAULT_LOSSES = {'train': float('inf'), 'test': float('inf')}
    DEFAULT_ACCURACIES = {'train': 0, 'test': 0}

    def process_single_state(args):
        args = args + [target_balls]

        try:
            result, output = safe_exec_function(learned_function, args)
            if result is None or np.isnan(np.sum(result)) or np.isinf(np.sum(result)):
                return None, output
            result = np.maximum(result, 0)  
            result_sum = np.sum(result)
            result = result / result_sum if result_sum > 0 else np.zeros_like(result)
            return result, None
        except Exception as e:
            return None, str(e)

    def process_dataset(dataset_subset):
        if evolve_type == 'value':
            starting_states = dataset_subset['starting_state']
            shots = dataset_subset['shot']
            states = dataset_subset['states']
            args_list = list(zip(starting_states, shots, states))
        elif evolve_type == 'difficulty':
            states = dataset_subset['states']
            shots = dataset_subset['shot']
            events = dataset_subset['event']
            args_list = list(zip(states, shots, events))
        
        if threading_eval:
            return process_dataset_threaded(args_list)
        else:
            return process_dataset_single(args_list)

    def process_dataset_single(args_list):
        results = []
        total_time = 0
        
        for args in args_list:
            t0 = time.time()
            result, output = process_single_state(list(args))
            total_time += time.time() - t0
            
            if result is None:
                return None, output, total_time
            results.append(result)
        
        return np.array(results), None, total_time

    def process_dataset_threaded(args_list):
        def process_chunk(chunk, chunk_start_index):
            results = []
            chunk_time = 0
            for i, args in enumerate(chunk):
                t0 = time.time()
                result, output = process_single_state(list(args))
                t1 = time.time() - t0
                chunk_time += t1
                if result is None:
                    return None, output, chunk_time
                results.append((chunk_start_index + i, result))
            return results, None, chunk_time

        chunk_size = ceil(len(args_list) / N_THREADS)
        chunks = [args_list[i:i+chunk_size] for i in range(0, len(args_list), chunk_size)]
        
        all_results = OrderedDict()
        total_chunk_time = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk, i * chunk_size): i 
                               for i, chunk in enumerate(chunks)}
            
            try:
                for future in concurrent.futures.as_completed(future_to_chunk, timeout=TIMEOUT):
                    chunk_results, output, chunk_time = future.result(timeout=TIMEOUT)
                    total_chunk_time = max(total_chunk_time, chunk_time)
                    if chunk_results is None:
                        return None, output, total_chunk_time
                    for idx, result in chunk_results:
                        all_results[idx] = result
            except concurrent.futures.TimeoutError:
                executor.shutdown(wait=False)
                return None, f"Evaluation timed out after {TIMEOUT} seconds", total_chunk_time
            except Exception as e:
                executor.shutdown(wait=False)
                return None, f"An error occurred: {str(e)}", total_chunk_time
            
        if len(all_results) != len(args_list):
            return None, "Incomplete results", total_chunk_time
        
        return np.array([all_results[i] for i in range(len(all_results))]), None, total_chunk_time

    # Process train and test datasets
    for subset in ['train', 'test']:
        y_hat, output, processing_time = process_dataset(dataset[subset])
        if y_hat is None:
            return DEFAULT_LOSSES, DEFAULT_ACCURACIES, output, -1
        
        if evolve_type == 'value':
            target = dataset[subset]['visit_distributions']
            losses[subset] = cross_entropy_loss(target, y_hat)
            accuracies[subset] = np.mean(np.argmax(y_hat, axis=1) == np.argmax(target, axis=1))
        else:
            y_hat = y_hat.flatten()
            target = dataset[subset]['difficulty'] 
            losses[subset] = mse_loss(target, y_hat)
            accuracies[subset] = 1 

        if np.isnan(losses[subset]):
            losses[subset] = float('inf')

    total_samples = sum(len(dataset[subset]['states']) for subset in ['train', 'test'])
    avg_time_per_sample = processing_time / total_samples

    return losses, accuracies, None, avg_time_per_sample

class ShotNode():
    '''
    Node in shot tree, contains:
        - shot params
        - new state
        - event list 
        - description of final state
        - children
        - parent
    '''
    def __init__(self):
        self.shot_params : dict = {}

        ### States before and after simulation
        self.start_state : State = None
        self.end_state : State = None

        ### All events that occurred in the shot 
        self.events_outcome : List[Event] = []
        ### Events that the LM Agent intended to occur in the shot
        self.events_intended : List[Event] = []
        ### Events that occurred in the shot and were intended to occurr
        self.cooccurrence_list : List[Event] = []
        ### Description of the final state that actually occurred
        self.final_state_true_description = ""

        self.children : List[ShotNode] = []
        self.parent : ShotNode = None
        
        ### Goal given by the agent
        self.goal : str = ""

        ### Difficulty found by sampling around the shot parameters
        self.difficulty : float = 0
        ### Rating of the shot based on the bayesian optimiser
        self.shot_search_rating : float = 0
        ### Rating of the shot based on the objective value of the state
        self.objective_value_rating : float = 0
        ### Standard deviation of the shot rating from the bayesian optimiser
        self.std_dev : float = 0
        ### Final rating of the node, combining all other ratings
        self.node_rating : float = 0

    def __str__(self):
        return f"""
        ShotNode:
            - Goal: {self.goal}
            - Start State: 
                {self.start_state}
            - End State: 
                {self.end_state}
            - Shot params: {self.shot_params}
            - Node Rating: {self.node_rating}
                - Difficulty: {self.difficulty}
                - Std Dev: {self.std_dev}
                - Shot Rating: {self.shot_search_rating}
                - Objective Value Rating: {self.objective_value_rating}
            - Final state true: {self.final_state_true_description}
            - Events intended: {self.events_intended}
            - Events outcome: {self.events_outcome}
            - Cooccurrence list: {self.cooccurrence_list}
        """
    
    def __repr__(self) -> str:
        return str(self)

    def to_json(self) -> dict:

        if self.start_state and isinstance(self.start_state, State):
            self.start_state = self.start_state.to_json()
        if self.end_state and isinstance(self.end_state, State):
            self.end_state = self.end_state.to_json()

        return {
            "goal": self.goal,
            "start_state": self.start_state,
            "end_state": self.end_state,
            "shot_params": self.shot_params,
            "node_rating": self.node_rating,
            "std_dev": self.std_dev,
            "difficulty": self.difficulty,
            "shot_rating": self.shot_search_rating,
            "objective_value_rating": self.objective_value_rating,
            "final_state_true_description": self.final_state_true_description,
            "events_intended": [str(e) for e in self.events_intended],
            "events_outcome":  [str(e) for e in self.events_outcome],
            "cooccurrence_list":  [str(e) for e in self.cooccurrence_list],
            "num_children": len(self.children)
        }
    
    @staticmethod
    def from_json(json) -> 'ShotNode':
        shot = ShotNode()
        shot.goal = json["goal"]
        shot.start_state = State.from_json(json["start_state"]) if json["start_state"] is not None else None
        shot.end_state = State.from_json(json["end_state"]) if json["end_state"] is not None else None
        shot.shot_params = json["shot_params"]
        shot.node_rating = json["node_rating"]
        shot.std_dev = json["std_dev"]
        shot.difficulty = json["difficulty"]
        shot.shot_search_rating = json["shot_rating"]
        shot.objective_value_rating = json["objective_value_rating"]
        shot.final_state_true_description = json["final_state_true_description"]
        shot.events_intended = [e for e in json["events_intended"]]
        shot.cooccurrence_list = [e for e in json["cooccurrence_list"]]
        shot.num_children = json["num_children"]
        return shot

class Trajectory():
    """A trajectory is a sequence of shots that are intended to achieve a goal. The trajectory is rated based on the ratings of the individual shots. The goal is given by a natural language description. The trajectory rating is the average of the shot ratings. The trajectory can be saved to a JSON file and visualized as a GIF.
    """
    def __init__(self):
        self.shots : List[ShotNode] = []
        self.goal : str = ""
        self.trajectory_rating : float = 0
        self.message = ""

    def add_shot(self, shot : ShotNode) -> None:
        self.shots.append(shot)

    def get_rating(self) -> float:

        rating = 0

        for s in self.shots:
            rating += s.node_rating

        return rating / len(self.shots)

    def to_json(self) -> dict:
        return {
            "trajectory_goal": self.goal,
            "trajectory_shots": [shot.to_json() for shot in self.shots],
            "trajectory_rating": self.get_rating() if self.trajectory_rating == 0 else self.trajectory_rating
        }

    @staticmethod
    def from_json(json) -> 'Trajectory':
        traj = Trajectory
        traj.goal = json["trajectory_goal"]
        traj.shots = [ShotNode.from_json(s) for s in json["trajectory_shots"]]
        traj.trajectory_rating = json["trajectory_rating"]
        return traj

def make_gif(trajectory, file_name, pool):
    """Create a GIF visualization of the provided trajectory by saving a GIF for each shot in the trajectory.

    Args:
        trajectory (_type_): The trajectory to visualize
        file_name (_type_): The base file name for the GIFs
        pool (_type_): The pool environment
    """
    file_name = file_name.split(".")[0]
    for idx, shot_node in enumerate(trajectory.shots[1:]):
        save_path = f"{file_name}--shot_{idx}.gif"
        if not os.path.exists(save_path):
            state = shot_node.start_state
            if isinstance(state, dict):
                state = State().from_json(state)
            params = shot_node.shot_params
            pool.save_shot_gif(state, params, f"{file_name}--shot_{idx}.gif")   

def save_trajectory(trajectory, idx, pool, save_dir):
    """Save a trajectory to a JSON file. Include a GIF visualization of each shot in the trajectory.

    Args:
        trajectory (_type_): The trajectory to save
        idx (_type_): The index of the trajectory
        pool (_type_): The pool environment
        save_dir (_type_): The directory to save the trajectory
    """
        
    data = trajectory.to_json()
    file_name = f"{save_dir}/trajectory_{idx}.json"

    with open(file_name, "w") as f:
        json.dump(data, f, indent=4)
    
    make_gif(trajectory, file_name, pool)

class Image():
    """A class to represent an image as a base64 string and a tensor. The image can be saved to a file or displayed in a window.
    """
    def __init__(self, image_base64, image_tensor) -> None:
        """Initialize the Image object with a base64 string and a tensor.

        Args:
            image_base64 (_type_): The base64 string representation of the image
            image_tensor (_type_): The tensor representation of the image
        """
        self.image_base64 = image_base64
        self.image_tensor = image_tensor

    def save_image(self, filename : str) -> None:
        """Save the image to a file.

        Args:
            filename (str): The name of the file to save the image to
        """
        image_data = base64.b64decode(self.image_base64)
        with open(filename, "wb") as f:
            f.write(image_data)

    def show_image(self) -> None:
        """Display the image in a window.
        """
        PIL.Image.open(io.BytesIO(base64.b64decode(self.image_base64))).show()

class EventType:
    """Enumeration of event types that can occur in a pool game.
    """
    NULL = -2
    ERROR = -1
    STICK_BALL = 0
    BALL_POCKET = 1
    BALL_BALL_COLLISION = 2
    BALL_CUSHION_COLLISION = 3
    BALL_STOP = 4

class Event():
    """A class to represent an event that occurs in a pool game. The event is defined by an event type, a set of arguments, and a position on the board.
    """

    VALID_BALLS = ["cue", "blue", "red", "yellow", "green", "black", "pink"]
    VALID_BALLS = VALID_BALLS + [str(i) for i in range(1,16)]
    VALID_POCKETS = ["rt", "lt", "rb", "lb", "rc", "lc"]

    POCKET_POSITIONS = {
        "rt": (1.0, 2.0),
        "lt": (0.0, 2.0),
        "rb": (1.0, 0.0),
        "lb": (0.0, 0.0),
        "rc": (1.0, 1.0),
        "lc": (0.0, 1.0),
    }

    DISTANCE_THRESHOLD = 0.0

    def __init__(self) -> None:
        self.encoding : str = ""
        self.event_type : EventType = EventType.NULL
        self.arguments : Tuple[str, str] = ("","")
        self.pos : Tuple[float, float] = None

    @staticmethod
    def distance(event1 : 'Event', event2 : 'Event') -> float:
        """Calculate the distance between two events and returns a value for how close the events are. Higher is better as we are adding this value to the bayes opt.

        Args:
            event1 (Event): The first event
            event2 (Event): The second event

        Returns:
            float: The distance between the two events
        """

        ### Returning a value for how close the events are, higher is better as we are adding this value to the bayes opt
        ### R = eps ** dist, where eps is a small number and dist is the distance between the two events
        ### Max 0.9 so that a near miss is never chosen over the actual event
        def reward(dist):
            return 0.1 ** dist - 0.1
        
        # Pocketing balls events are close to cushion events that have a position 
        if event1.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event2.event_type == EventType.BALL_CUSHION_COLLISION and event2.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event1.arguments[1]]) - np.array(event2.pos))
            return reward(dist)
        if event2.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event1.event_type == EventType.BALL_CUSHION_COLLISION and event1.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event2.arguments[1]]) - np.array(event1.pos))
            return reward(dist)
        
        # Pocketing balls events are close to ball stop events that have a position
        if event1.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event2.event_type == EventType.BALL_STOP and event2.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event1.arguments[1]]) - np.array(event2.pos))
            return reward(dist)
        if event2.event_type == EventType.BALL_POCKET and event1.arguments[1] != "" and event1.event_type == EventType.BALL_STOP and event1.pos is not None:
            dist = np.linalg.norm(np.array(Event.POCKET_POSITIONS[event2.arguments[1]]) - np.array(event1.pos))
            return reward(dist)

        return 0.0
        
    @staticmethod
    def parse_position(pos : str) -> Tuple[float, float]:
        """Parse a position string, like "(0.1, 0.2)" into a tuple of floats.

        Args:
            pos (str): The position string to parse

        Raises:
            ValueError: If the position string is invalid

        Returns:
            Tuple[float, float]: The position as a tuple of floats
        """

        if pos is None:
            return None

        try:
            pos = pos.replace("(", "")
            pos = pos.replace(")", "")
            pos = pos.split(",")
            return (float(pos[0]), float(pos[1]))
        except:
            print(f"Invalid position string: {pos}")
            raise ValueError

    @staticmethod
    def get_closest_event(event : 'Event', events : List['Event']) -> 'Event':
        """Get the event in the list of events that is closest to the given event, using the Event.distance function.

        Args:
            event (Event): The event to compare against
            events (List[Event]): The list of events to search

        Returns:
            Event: The closest event in the list
        """
        closest_event = None
        min_distance = np.inf

        for e in events:
            distance = Event.distance(event, e)
            if distance > Event.DISTANCE_THRESHOLD and distance < min_distance:
                min_distance = distance
                closest_event = e

        return closest_event

    @staticmethod
    def get_cooccurrence(events1 : List['Event'], events2 : List['Event']) -> List['Event']:
        """Get the list of events that occur in both lists of events.

        Args:
            events1 (List[Event]): The first list of events
            events2 (List[Event]): The second list of events

        Returns:
            _type_: The list of events that occur in both lists
        """
        cooccurrence = []
        for event1 in events1:
            for event2 in events2:
                if event1 == event2:
                    cooccurrence.append(event1)

        return cooccurrence

    @staticmethod
    def ball_collision(ball1 : str, ball2 : str = "", pos : str = "") -> 'Event':
        """Create a ball collision event between two balls.

        Args:
            ball1 (str): The first ball in the collision
            ball2 (str, optional): The second ball in the collision, if blank then event is general. Defaults to "".
            pos (str, optional): The position of the event. Defaults to "".

        Returns:
            Event: The ball collision event
        """
        event = Event()

        if ball1 not in Event.VALID_BALLS:
            return event
        
        event.event_type = EventType.BALL_BALL_COLLISION
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball2 == "":
            event.arguments = (ball1, "")
            event.encoding = f"ball-ball-{ball1}"
            return event
        
        if ball2 not in Event.VALID_BALLS:
            return event

        event.arguments = (ball1, ball2)
        event.encoding = f"ball-ball-{ball1}-{ball2}"
        return event
    
    @staticmethod
    def ball_pocket(ball : str, pocket : str = "", pos : str = "") -> 'Event':
        """Create a ball pocket event. If pocket is not specified then it is a generic pocket.
        Balls = ["cue", "blue", "red", "yellow"]
        Pockets = ["rt", "lt", "rb", "lb", "rc", "lc"]

        Args:
            ball (str): The ball that was pocketed
            pocket (str, optional): The pocket the ball was pocketed in. Defaults to "".
            pos (str, optional): The position of the event. Defaults to "".

        Returns:
            Event: The ball pocket event
        """
        event = Event()

        if ball not in Event.VALID_BALLS:
            return event
        
        event.event_type = EventType.BALL_POCKET
        event.pos = Event.parse_position(pos) if pos != "" else None

        if pocket == "":
            event.arguments = (ball, "")
            event.encoding = f"ball-pocket-{ball}"
            return event
        
        if pocket not in Event.VALID_POCKETS:
            return event

        event.arguments = (ball, pocket)
        event.encoding = f"ball-pocket-{ball}-{pocket}"
        return event
    
    @staticmethod
    def ball_cushion(ball : str, c_id : str = "", pos : str = "") -> 'Event':
        """Create a ball cushion collision event.

        Args:
            ball (str): The ball that collided with the cushion
            c_id (str, optional): The cushion ID that the ball collided with. Defaults to "".
            pos (str, optional): The position of the event. Defaults to "".

        Returns:
            Event: The ball cushion collision event
        """
        event = Event()

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.BALL_CUSHION_COLLISION
        event.pos = Event.parse_position(pos) if pos != "" else None

        if c_id == "":
            event.arguments = (ball,"")
            event.encoding = f"ball-cushion-{ball}"
            return event

        # TODO: cant find where a list of cushion ids could be found
        # if c_id not in Event.VALID_CUSHIONS:
        #     return event
        
        event.arguments = (ball, c_id)
        event.encoding = f"ball-cushion-{ball}-{c_id}"

        return event

    @staticmethod
    def ball_stop(ball : str, pos : str = "") -> 'Event':
        """Create a ball stop event.

        Args:
            ball (str): The ball that stopped
            pos (str, optional): The position of the event. Defaults to "".

        Returns:
            Event: The ball stop event
        """
        event = Event()
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.BALL_STOP
        event.arguments = (ball,"")
        event.encoding = f"ball-stop-{ball}"
        return event
    
    @staticmethod
    def stick_ball(ball : str, pos : str = "") -> 'Event':
        """Create a cue ball stick event.

        Args:
            ball (str): The cue ball
            pos (str, optional): The position of the event. Defaults to "".

        Returns:
            Event: The cue ball stick event
        """
        event = Event()
        event.pos = Event.parse_position(pos) if pos != "" else None

        if ball not in Event.VALID_BALLS:
            return event

        event.event_type = EventType.STICK_BALL
        event.arguments = (ball,"")
        event.encoding = f"stick-ball-{ball}"
        return event

    @staticmethod
    def null_event() -> 'Event':
        """Create a null event, used for initialization.

        Returns:
            Event: The null event
        """
        return Event()

    @staticmethod
    def from_encoding(encoding_str : str, pos : tuple = None) -> 'Event':
        """Create an event from an encoding string.

        Args:
            encoding_str (str): The encoding string

        Returns:
            Event: The event created from the encoding string
        """

        def error():
            print(f"Invalid event encoding: {encoding_str}")
            return Event()
        
        def encoding_index(encoding):
            if len(encoding) == 0:
                return ""
            return encoding.pop(0)

        encoding = encoding_str.strip()
        encoding = encoding.replace("\\n", "")
        encoding = encoding.replace(" ", "")
        encoding = encoding.strip()

        encoding = encoding.lower()
        encoding = encoding.split("-")

        e_type = encoding_index(encoding) + "-" + encoding_index(encoding)
        args = [
            encoding_index(encoding),
            encoding_index(encoding),
        ]

        if pos and not isinstance(pos, str):
            pos = f"({pos[0]:.2f},{pos[1]:.2f})"

        if pos is None:
            pos = ""

        if args == ["", ""]:
            return error()
        
        if e_type == "stick-ball":
            return Event.stick_ball(args[1], pos)
        
        elif e_type == "ball-pocket":
            return Event.ball_pocket(args[0], args[1], pos)
            

        elif e_type == "ball-ball":
            return Event.ball_collision(args[0], args[1], pos)

        elif e_type == "ball-cushion":
            return Event.ball_cushion(args[0], args[1], pos)
        
        elif e_type == "ball-stop":
            return Event.ball_stop(args[0], pos)

        return Event()

    def to_encoding(self) -> str:
        return self.encoding
    
    def __eq__(self, other : 'Event') -> bool:
        """Check if two events are equal. For generic events, only the ball is compared. For specific events, the ball and args are compared.

        Args:
            other (Event): The other event to compare

        Returns:
            bool: True if the events are equal, False otherwise
        """

        if not isinstance(other, Event):
            return False
        
        if self.encoding == other.encoding:
            return True

        # Check for generic events
        either_generic = self.arguments[1] == "" or other.arguments[1] == ""
        if either_generic and self.event_type == other.event_type:
            return self.arguments[0] == other.arguments[0]

        return False

    def to_json(self) -> dict:
        return {
            "encoding": self.encoding,
            "pos": self.pos
        }
    
    def from_json(json) -> 'Event':
        encoding = json["encoding"]
        pos = json["pos"]
        return Event.from_encoding(encoding, pos)
    
    def __str__(self) -> str:
        return f"Event({self.encoding}, {self.pos})"
    
    def __repr__(self) -> str:
        return f"Event({self.encoding}, {self.pos})"

class State():
    """A class to represent the state of a pool game. The state is defined by the positions of the balls on the table and the parameters of the shot that led to the state.
    """
    def __init__(self, positions : Dict[str, List[float]] = None, params : Dict[str, float] = None, random : bool = False, num_ball_level : int = 3):
        """Initialize the State object with the positions of the balls and the shot parameters.

        Args:
            positions (Dict[str, List[float]], optional): The positions of the balls. Defaults to None.
            params (Dict[str, float], optional): The shot parameters. Defaults to None.
            random (bool, optional): Whether to randomize the state. Defaults to False.
            num_ball_level (int, optional): The number of balls for each player.
        """

        self.num_ball_level = num_ball_level

        # If ball is potted (i.e. at infinity) then remove it from the state rather than setting it to infinity
        if positions is not None:
            for k, v in positions.items():
                if isinstance(v, str):
                    del positions[k]

        self.ball_positions : Dict[str, List[float]] = positions if positions else {
            "cue": [0.5, 0.5],
            "red": [0.25, 1.5],
            "yellow": [0.5, 1.5],
            "blue": [0.75, 1.5],
        }
        self.pocket_positions : Dict[str, List[float]] = {
            'lt': [0.0,2.0],
            'rt': [1.0,2.0],
            'lc': [0.0,1.0],
            'rc': [1.0,1.0],
            'lb': [0.0,0.0],
            'rb': [1.0,0.0],
        }
        self.params : Dict[str, float] = params
        self.ball_radius : float = 0.028575
        self.table_width_height : Tuple[float, float] = (0.9906, 1.9812)

        if random:
            self.randomize()

        # For 2 player games
        self.current_player = "one"

    def angle_between_balls(self, ball1 : str, ball2 : str) -> float:
        """Calculate the angle between two balls.

        Args:
            ball1 (str): The first ball
            ball2 (str): The second ball

        Returns:
            float: The angle between the two balls
        """
        if not ball1 in self.ball_positions.keys() or not ball2 in self.ball_positions.keys():
            return 0.0
        
        ball1_pos = self.ball_positions[ball1]
        ball2_pos = self.ball_positions[ball2]

        return np.arctan2(ball2_pos[1] - ball1_pos[1], ball2_pos[0] - ball1_pos[0])

    def get_ball_position(self, ball : str) -> np.ndarray:
        """Get the position of a ball.

        Args:
            ball (str): The ball to get the position of

        Returns:
            np.ndarray: The position of the ball
        """

        if not ball in self.ball_positions.keys():
            return np.array([np.inf, np.inf], dtype=float)
        
        pos = self.ball_positions[ball]

        if isinstance(pos, str):
            return np.array([np.inf, np.inf], dtype=float)
        
        return np.array(pos, dtype=float)
    
    def get_pocket_position(self, pocket : str) -> np.ndarray:
        """Get the position of a pocket.

        Args:
            pocket (str): The pocket to get the position of

        Returns:
            np.ndarray: The position of the pocket
        """

        if not pocket in self.pocket_positions.keys():
            return np.array([np.inf, np.inf], dtype=float)

        return np.array(self.pocket_positions[pocket], dtype=float)

    def all_pocket_keys(self) -> List[str]:
        """Get the keys of all pockets.

        Returns:
            List[str]: The keys of all pockets
        """
        return list(self.pocket_positions.keys())

    def all_ball_keys(self) -> List[str]:
        """Get the keys of all balls.

        Returns:
            List[str]: The keys of all balls
        """
        return list(self.ball_positions.keys())

    def angle_to_pocket(self, ball : str, pocket : str) -> float:
        """Calculate the angle between a ball and a pocket.

        Args:
            ball (str): The ball ID
            pocket (str): The pocket ID

        Returns:
            float: The angle between the ball and the pocket
        """

        if not ball in self.ball_positions.keys() or not pocket in self.pocket_positions.keys():
            return 0.0
        ball_pos = self.ball_positions[ball]
        pocket_pos = self.pocket_positions[pocket]
        return np.arctan2(pocket_pos[1] - ball_pos[1], pocket_pos[0] - ball_pos[0])

    def line_of_sight(self, start: List[float], end: List[float]) -> bool:
        """Calculate if there is a line of sight between two points, i.e. if there are no balls in the way.

        Args:
            start (List[float]): The start point
            end (List[float]): The end point

        Returns:
            bool: True if there is a line of sight, False otherwise
        """

        if isinstance(start[0], str) or isinstance(end[0], str):
            return False

        if np.isinf(start).any() or np.isinf(end).any():
            return False

        # Get vector from start to end
        direction = [end[0] - start[0], end[1] - start[1]]
        distance = (direction[0]**2 + direction[1]**2)**0.5
        
        if distance == 0:
            return True
        
        # Normalize direction vector
        direction = [d/distance for d in direction]
        
        BALL_RADIUS_SQ = 0.028575 * 0.028575

        # For each ball, check if it intersects with the line
        for ball_pos in self.ball_positions.values():

            if isinstance(ball_pos[0], str) or isinstance(ball_pos[0], str):
                continue

            if np.isinf(ball_pos).any() or np.isinf(ball_pos).any():
                continue

            # Skip if ball_pos already within ball radius of start or end
            if (ball_pos[0] - start[0])**2 + (ball_pos[1] - start[1])**2 < BALL_RADIUS_SQ:
                continue
            if (ball_pos[0] - end[0])**2 + (ball_pos[1] - end[1])**2 < BALL_RADIUS_SQ:
                continue

            # Vector from start to ball
            to_ball = [ball_pos[0] - start[0], ball_pos[1] - start[1]]
            
            # Project ball onto line of sight
            dot_product = to_ball[0]*direction[0] + to_ball[1]*direction[1]
            
            # Find closest point on line to ball
            closest_point = [
                start[0] + direction[0]*max(0, min(dot_product, distance)),
                start[1] + direction[1]*max(0, min(dot_product, distance))
            ]
            
            # Calculate distance from ball to line
            ball_to_line = [
                ball_pos[0] - closest_point[0],
                ball_pos[1] - closest_point[1]
            ]
            perpendicular_distance_sq = (ball_to_line[0]**2 + ball_to_line[1]**2)
                        
            if perpendicular_distance_sq < BALL_RADIUS_SQ:
                # Check if the ball is actually between start and end
                if 0 <= dot_product <= distance:
                    return False
        
        return True

    def get_id(self) -> str:
        """An attempt to get a unique ID for the state, to be used for the agent visualisation.

        Returns:
            str: The unique ID for the state
        """
        val = abs(self.ball_positions["cue"][0] + self.ball_positions["cue"][1]) ** 2
        return f"{val:.8f}"

    def is_potted(self, ball : str) -> bool:
        """Check if a ball is potted by seeing if its position is infinity.

        NOTE: THIS IS FOR LEARNED FUNCTION.

        Args:
            ball (str): The ball to check

        Returns:
            bool: True if the ball is potted, False otherwise
        """

        if not ball in self.ball_positions.keys():
            return True
        
        if isinstance(self.ball_positions[ball][0], str) or np.isinf(self.ball_positions[ball]).any():
            return True

        return self.ball_positions[ball][0] == np.inf and self.ball_positions[ball][1] == np.inf

    def from_board_state(self, board_state) -> None:
        """Create a state from a board state. The board state is a dictionary with the positions of the balls, found from the PoolTool simulation.

        Args:
            board_state (_type_): The board state to create the state from
        """
        self.params = None

        positions : dict = board_state["balls"]
        self.ball_positions = {}
        for k in positions.keys():
            self.ball_positions[k] = positions.get(k, [np.inf, np.inf])
        self.num_ball_level = (len(self.ball_positions.keys()) - 1) // 2
    
    def get_state_description(self) -> str:
        """Generate a natural language description of the state, based on the positions of the balls.

        Returns:
            str: The natural language description of the state
        """
        caption = ''
        for key, el in self.ball_positions.items():

            # Check if potted
            if el[0] == np.inf and el[1] == np.inf:
                caption += f'{key} ball is potted. '
                continue

            quadrant = ''
            if el[0] < 0.33:
                quadrant += 'near the left-{} pocket'
            elif el[0] > 0.33 and el[0] < 0.66:
                quadrant += 'in between the left-{} and right-{} pockets'
            else:
                quadrant += 'near the right-{} pocket'
            if el[1] < 0.66:
                column = 'bottom'
            elif el[1] > 0.66 and el[1] < 1.33:
                column = 'center'
            else:
                column = 'top'

            if 'in between' in quadrant:
                quadrant = quadrant.format(column, column)
            else:
                quadrant = quadrant.format(column)
            caption += f'{key} ball is {quadrant}. '
        return caption

    def set_params(self, params) -> 'State':
        """Set the parameters of the state.

        Args:
            params (_type_): The shot parameters that led to the state

        Returns:
            State: The state with the parameters set
        """
        self.params = params
        self.static = False
        return self
    
    def copy(self) -> 'State':
        """Create a shallow copy of the state.

        Returns:
            State: The copy of the state
        """
        state = State(self.ball_positions.copy(), self.params.copy() if self.params else None)
        state.current_player = self.current_player
        return state
    
    def randomize(self) -> 'State':
        """Randomize the positions of the balls on the table. If the balls are overlapping, randomize again.

        Returns:
            State: The state with the balls randomized
        """

        R = self.ball_radius * 1.1 # Add a small buffer to ensure balls are not overlapping cushions 
        for ball in self.ball_positions:
            self.ball_positions[ball] = [
                np.random.uniform(R, self.table_width_height[0] - R), 
                np.random.uniform(R, self.table_width_height[1] - R)
            ]

        # If balls are overlapping then randomize again
        if self.balls_overlapping():
            self.randomize()

        return self
    
    def balls_overlapping(self) -> bool:
        """Check if any balls are overlapping.

        Returns:
            bool: True if balls are overlapping, False otherwise
        """
        for ball1 in self.ball_positions:
            for ball2 in self.ball_positions:
                if ball1 != ball2:
                    if np.linalg.norm(np.array(self.ball_positions[ball1]) - np.array(self.ball_positions[ball2])) < 2*self.ball_radius:
                        return True
        return False

    def to_json(self) -> dict:
        
        for k, v in self.ball_positions.items():
            if v[0] == np.inf:
                self.ball_positions[k] = ["infinity", "infinity"]

        return {
            "positions": self.ball_positions,
            "params": self.params
        }
    
    @staticmethod
    def from_json(json_dict : dict) -> 'State':
        return State(json_dict["positions"], json_dict["params"])
    
    def get_balls(self) -> str:
        """Return a string of the ball IDs that are currently on the table. This is used to provide information on the current state to the LLM agent.

        Returns:
            str: The string of ball IDs
        """

        current_balls = []

        for key, el in self.ball_positions.items():

            # Check if potted
            if el[0] == np.inf and el[1] == np.inf:
                continue

            current_balls.append(key)

        return "Ball IDs: " + ", ".join(current_balls)
    
    def full_randomize(self) -> 'State':
        """Randomize the positions of the balls on the table. If the balls are overlapping, randomize again. Then, remove 0, 1, or 2 of the red, blue, yellow balls, and if the green, pink, black balls are there then remove 0, 1, or 2 of them too. This is to ensure the MCTS data generation has good diversity.

        Returns:
            State: The state with the balls randomized
        """
        self.randomize()
        
        # Remove 0,1,2 of the red, blue, yellow balls, and if the green, pink, black balls are there then remove 0,1,2 of them too 
        # This is to ensure the MCTS data generation has good diversity
        remove_amount1 = np.random.randint(0, 3)
        remove_amount2 = np.random.randint(0, 3)

        player1_balls = ["red", "blue", "yellow"]
        random.shuffle(player1_balls)
        to_remove = player1_balls[:remove_amount1]

        if len(self.ball_positions.keys()) > 3:
            player2_balls = ["green", "pink", "black"]
            random.shuffle(player2_balls)
            to_remove += player2_balls[:remove_amount2]

        for ball in to_remove:
            self.ball_positions[ball] = [np.inf, np.inf]

        return self

    
class TwoPlayerState(State):
    """This is a special state setup where the balls are split between two players. 
    """

    def __init__(self, random : bool = False, num_ball_level : int = None):
        super().__init__(random=random, num_ball_level=num_ball_level)

        self.ball_positions = {
            "cue": [0.5, 0.5],
            "red": [0.25, 1.75],
            "yellow": [0.25, 1.5],
            "blue": [0.25, 1.25],
            "green": [0.75, 1.75],
            "black": [0.75, 1.5],
            "pink": [0.75, 1.25],
        }

        if not num_ball_level is None:
            self.ball_positions = {
                "cue": [0.5, 0.5]
            }
            for i in range(1, num_ball_level+1):
                self.ball_positions[str(i)] = [0.25, 1.75 - 0.1*(i - 1)]
            for i in range(9, num_ball_level+9):
                self.ball_positions[str(i)] = [0.75, 1.75 - 0.1*(i - 1)]

        if random:
            self.randomize()
        

def draw_pool_table(board_state, image_width=122, image_height=244) -> Image :
    board_state = board_state.copy()

    # Fixed hole positions
    holes = [
        (0.0, 0.0), 
        (0.0, 2.0), 
        (-0.025, 1,0),
        (1.0, 0.0),
        (1.0, 2.0),
        (1.025, 1,0),
    ]

    # Change cue ball color to white
    if "cue" in board_state:
        board_state["white"] = board_state.pop("cue")
    
    # Create a blank white image
    image = PIL.Image.new("RGB", (image_width, image_height), "green")
    draw = ImageDraw.Draw(image)
    
    # Convert coordinates from 0-1 in X and 0-2 in Y to pixel coordinates
    def convert_coordinates(position):
        x = int(position[0] * image_width)
        y = int((2.0 - position[1]) * image_height / 2) # FIX: Flip Y axis as image was upside down
        return (x, y)
    
    # Draw holes
    HOLE_SCALE = 0.05
    hole_radius = int(image_width * HOLE_SCALE)  # proportional to image width
    for hole_position in holes:
        hole_position_pixel = convert_coordinates(hole_position)
        draw.ellipse((hole_position_pixel[0] - hole_radius, hole_position_pixel[1] - hole_radius,
                      hole_position_pixel[0] + hole_radius, hole_position_pixel[1] + hole_radius),
                     fill="black")
    
    # Draw balls
    BALL_SCALE = 0.03
    ball_radius = int(image_width * BALL_SCALE)  # proportional to image width
    for ball_color, ball_position in board_state.items():
        clipped_ball_position = [0, 0]
        clipped_ball_position[0] = np.clip(float(ball_position[0]), -100, 100)
        clipped_ball_position[1] = np.clip(float(ball_position[1]), -100, 100)
        ball_position_pixel = convert_coordinates(clipped_ball_position)
        draw.ellipse((ball_position_pixel[0] - ball_radius, ball_position_pixel[1] - ball_radius,
                      ball_position_pixel[0] + ball_radius, ball_position_pixel[1] + ball_radius),
                     fill=ball_color)
        
    # # Pad image with white space, to make a square image (2*width, height)
    # image = image.crop((0, 0, 2*image_width, image_height))
    
    # Convert image to base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Convert image to numpy array 
    image = np.array(image).reshape(1, 3, image_width, image_height)
    
    return Image(image_base64, image)

def get_image_util(state):
    return draw_pool_table(state.ball_positions)