import torch, random, logging, os, json, re, time, dotenv, dspy, subprocess, signal
import numpy as np
from datetime import datetime
import threading
import time
from queue import Queue
import traceback
import time
from functools import wraps

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich import box

from poolagent.path import ROOT_DIR

### Constants ###
GPU_SIZE = 40
MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2

### Experiment Classes to Override ###

class Task():
    """Object representing a task to be run in the experiment. Simply contains the necessary information on a task.
    """
    def __init__(self, description, models):
        self.model_id = None
        self.description = description
        self.models = models
        self.assigned_llms = {}

    def __str__(self):
        return self.description

    def __repr__(self):
        return self.description

class Experiment:
    """Object representing an experiment to be run. Need to override this and implement run_task to run your own experiments.
    """
    def __init__(self):
        self.tasks = []  
        self.results = {}
        self.current_tasks = {}

    def get_tasks_by_models(self, model_ids):
        tasks = []
        for task in self.tasks:

            if all( (model_id in model_ids or model_id is None) for model_id in task.models):
                tasks.append(task)

        return tasks
    
    def remove_task(self, task):
        if task in self.tasks:
            self.tasks.remove(task)

    def run_task(self, task, thread_id, timestamp, N=1, logger=None):
        raise NotImplementedError

### Profiling and Resource Management ###


class Profiler:
    def __init__(self):
        self.function_times = {}

    def profile(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            
            func_name = func.__name__
            execution_time = end_time - start_time
            
            if func_name not in self.function_times:
                self.function_times[func_name] = []
            self.function_times[func_name].append(execution_time)
            
            return result
        return wrapper

profiler = Profiler()

class LLM:
    def __init__(self, model_id, gpu_ids=[], timeout=5, total_gpus=1, gpu_size=40, temperature=0.0, max_tokens=512, repetition_penalty=1.1):
        self.total_gpus = total_gpus
        self.gpu_size = gpu_size
        self.model_id = model_id
        self.gpu_ids = gpu_ids
        self.timeout = timeout
        self.logger = None
        self.temperature = temperature
        self.process = None
        self.max_tokens = max_tokens
        self.repetition_penalty = repetition_penalty
        self.llm = self.setup_model_dspy()

    def setup_model_dspy(self):
        
        MAX_TOKENS = self.max_tokens
        REPETITION_PENALTY = self.repetition_penalty

        if not self.gpu_ids:

            if 'together/' in self.model_id:

                print(f"Loading Together model {self.model_id}")

                model_id = self.model_id.replace("together/", "")

                config = {}
                if 'llama' in model_id:
                    config['eos_token'] = "<|end_of_text|>"
                    config["pad_token"] = "<|reserved_special_token_0|>"

                llm = dspy.Together(
                    model=model_id,
                    max_tokens=MAX_TOKENS,
                    temperature=self.temperature,
                    repetition_penalty=REPETITION_PENALTY,
                    **config
                )
                llm.use_inst_template = False

                return llm

            if 'gpt' in self.model_id:
                # Azure OpenAI setup 
                dotenv.load_dotenv()
                assert os.getenv("API_KEY") is not None, "API_KEY not found in .env file, this is required for the Azure OpenAI API"
                assert os.getenv("API_BASE") is not None, "API_BASE not found in .env file, this is required for the Azure OpenAI API"
                api_key = os.getenv("API_KEY")
                base_url = os.getenv("API_BASE")
                api_version = '2024-06-01'

                config = {
                    "temperature": self.temperature,
                    "max_tokens": MAX_TOKENS,
                }

                azure_llm = dspy.AzureOpenAI(
                    model=self.model_id,
                    api_base=base_url,
                    api_version=api_version,
                    api_key=api_key,
                    **config
                )
                return azure_llm
        else:

            gpu_idx = [int(gpu_id.split("_")[1]) - 1 for gpu_id in self.gpu_ids]
            devices = ",".join(str(idx) for idx in gpu_idx)
            os.environ["CUDA_VISIBLE_DEVICES"] = devices
            print(f"Loading model {self.model_id} on GPUs {gpu_idx}")

            llm = dspy.HFModel(
                model = self.model_id,
                hf_device_map=f"auto",
                model_kwargs = {
                    "max_tokens": MAX_TOKENS,
                    "temperature": self.temperature,
                    "torch_dtype": torch.float16,                
                }
            )
            return llm
        
            # from dspy import HFClientVLLM

            # # vLLM setup
            # gpu_idx = [int(gpu_id.split("_")[1]) - 1 for gpu_id in self.gpu_ids]
            # devices = ",".join(str(idx) for idx in gpu_idx)
            # os.environ["CUDA_VISIBLE_DEVICES"] = devices

            # print(f"Starting vLLM server with model {self.model_id} on GPUs {devices}")

            # # Start vLLM server as a subprocess
            # command = f"python3 -m vllm.entrypoints.openai.api_server --model {self.model_id} --gpu_memory_utilization 0.90 --tensor_parallel_size {len(self.gpu_ids)} --port 800{gpu_idx[0]}"
            # self.process = subprocess.Popen(
            #     command,
            #     shell=True,
            #     stdout=subprocess.PIPE,
            #     stderr=subprocess.STDOUT,
            #     universal_newlines=True,
            #     preexec_fn=os.setsid
            # )

            # llm = HFClientVLLM(
            #     model=self.model_id,
            #     port=f"800{gpu_idx[0]}",
            #     model_type="chat",
            #     max_tokens=1024,
            #     temperature=self.temperature
            # )

            # # Wait for the server to start
            # loading = True
            # t0 = time.time()
            # while(loading):
                
            #     if self.process.poll() is not None:
            #         print(f"vLLM server failed to start. Exiting...")
            #         raise Exception(f"vLLM server failed to start for model {self.model_id} on GPUs {devices}")

            #     try:
            #         response = llm._generate(prompt=f'Test Prompt {random.random()}')
            #         assert len(response['choices']) > 0, "No response from vLLM server"
            #         loading = False
            #     except:
            #         print(f"Waiting for vLLM server to start... {time.time() - t0:.2f}s")
            #         time.sleep(self.timeout)
            
            # print("vLLM server started successfully")

            # return llm

    def delete(self):
        if self.process:
            # Terminate the vLLM server process and all its children
            os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            self.process = None
        
        # Clear the CUDA_VISIBLE_DEVICES environment variable
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        # Delete the LLM object
        del self.llm

class ResourceManager:
    def __init__(self, model_ids, gpu_ids, temperature=0.0):
        self.llm_instances = {model_id: [] for model_id in model_ids}
        self.inactive_models = set(model_ids)
        self.gpu_ids = gpu_ids
        self.temperature = temperature

    def get_model_size(self, model_id):
        match = re.search(r'-(\d+)b', model_id.lower())
        return int(match.group(1)) if match else 0

    def get_required_gpus(self, model_size):
        return 1 if model_size < MAX_MODEL_SIZE_FOR_SINGLE_GPU else (model_size - 1) // MAX_MODEL_SIZE_FOR_SINGLE_GPU + 1

    @profiler.profile
    def assign_gpu(self, model_id):
        if self.is_api_model(model_id):
            self.inactive_models.discard(model_id)
            llm = LLM(model_id, temperature=self.temperature)
            self.llm_instances[model_id].append({
                'gpus': None,
                'active': 0,
                'LLM': llm
            })
            return True

        model_size = self.get_model_size(model_id)
        required_gpus = self.get_required_gpus(model_size)
        available_gpus = self.get_available_gpus()

        if len(available_gpus) >= required_gpus:
            assigned_gpus = tuple(available_gpus[:required_gpus])
            self.inactive_models.discard(model_id)
            llm = LLM(model_id, assigned_gpus, total_gpus=len(self.gpu_ids), gpu_size=GPU_SIZE, temperature=self.temperature)
            self.llm_instances[model_id].append({
                'gpus': assigned_gpus,
                'active': 0,
                'LLM': llm
            })
            return True
        return False

    def is_api_model(self, model_id):
        return any([name in model_id for name in ['gpt', 'together']])

    def get_available_models(self):
        return [model_id for model_id, instances in self.llm_instances.items() 
                if any( instance['active']==0 for instance in instances)]

    def get_active_models(self):
        return [model_id for model_id, instances in self.llm_instances.items() if instances]

    def get_available_gpus(self):
        used_gpus = set()
        for instances in self.llm_instances.values():
            for instance in instances:
                if instance['gpus']:
                    used_gpus.update(instance['gpus'])
        all_gpus = set(self.gpu_ids)
        return list(all_gpus - used_gpus)

    def unassign_gpus(self, model_ids):
        for model_id in model_ids:
            for instance in self.llm_instances[model_id]:
                instance['LLM'].delete()
            self.llm_instances[model_id] = []
            self.inactive_models.add(model_id)

    def replace_models(self, old_model_ids, new_model_id=None):
        self.unassign_gpus(old_model_ids)
        if new_model_id is None:
            new_model_id = self.find_replacement_model(old_model_ids)
        if new_model_id:
            self.assign_gpu(new_model_id)
        return new_model_id

    def find_replacement_model(self, old_model_ids):
        if self.inactive_models:
            inactive_models = list(self.inactive_models - set(old_model_ids))
            return random.choice(inactive_models) if inactive_models else None
        active_models = self.get_active_models()
        # Return the model with the least number of instances
        return np.argmin([len(self.llm_instances[model_id]) for model_id in active_models])

    def get_inactive_llm_instance(self, model_id):
        instances = self.llm_instances.get(model_id, [])
        for instance in instances:
            if instance['active'] == 0 or self.is_api_model(model_id):
                instance['active'] += 1 
                return instance['LLM']
        
        # No inactive instances found, must return an active instance

        if len(instances) == 0:
            return None
        
        if len(instances) == 1:
            instances[0]['active'] += 1 
            return instances[0]['LLM']

        if len(instances) > 1:
            # Return least used instance
            instance_idx = np.argmin([inst['active'] for inst in instances])
            instances[instance_idx]['active'] += 1 
            return instances[instance_idx]['LLM'] 

        return None

    def release_llm(self, model_id, llm):
        if not llm:
            return True
        for instance in self.llm_instances.get(model_id, []):
            if instance['LLM'] == llm:
                instance['active'] -= 1 
                return True
        return False

### Experiment Manager ###

class ExperimentManager:
    def __init__(self, experiment_name, model_ids, gpu_ids, experiment, max_concurrent_threads=1, N=1, gpu_size=40, temperature=0.0):
        self.experiment_name = experiment_name
        self.max_concurrent_threads = max_concurrent_threads
        self.profiler = profiler
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.thread_loggers = {}
        self.logger = self.setup_logging()
        self.N = N
        self.print_console = Console()
        self.temperature = temperature

        global GPU_SIZE, MAX_MODEL_SIZE_FOR_SINGLE_GPU
        GPU_SIZE = gpu_size
        MAX_MODEL_SIZE_FOR_SINGLE_GPU = GPU_SIZE // 2

        self.results_dir = f"{ROOT_DIR}/experiments/{self.experiment_name}/logs/{self.timestamp}/"
        os.makedirs(self.results_dir, exist_ok=True)

        self.resource_manager = ResourceManager(model_ids, gpu_ids, temperature=self.temperature)
        self.experiment = experiment
        self.initial_num_tasks = len(self.experiment.tasks)
        self.active_threads = threading.Semaphore(max_concurrent_threads)
        self.task_queue = Queue()
        self.results = self.experiment.results

        self.wait_time = 5

        self.start_time = time.time()

    def setup_logging(self):
        log_dir = f"{ROOT_DIR}/experiments/{self.experiment_name}/logs/{self.timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{log_dir}/main.log"),
                logging.StreamHandler()
            ]
        )

        # Set up thread-specific logging
        for i in range(self.max_concurrent_threads):
            thread_logger = logging.getLogger(f"Thread-{i}")
            thread_logger.setLevel(logging.INFO)
            thread_logger.format = logging.Formatter('%(asctime)s - %(message)s')
            thread_logger.addHandler(logging.FileHandler(f"{log_dir}/thread_{i}.log"))
            thread_logger.addHandler(logging.StreamHandler())
            self.thread_loggers[i] = thread_logger

        return logging.getLogger()

    def thread_worker(self, thread_id):
        thread_logger = self.thread_loggers[thread_id]
        while True:
            task = self.task_queue.get()
            if task is None:
                break
            self.active_threads.acquire()
            self.run_task(task, thread_id, thread_logger)
            self.active_threads.release()
            self.task_queue.task_done()

    def update_results(self):
        if not os.path.exists(f"{self.results_dir}/tasks"):
            return

        for task_dir in os.listdir(f"{self.results_dir}/tasks"):
            task_key = task_dir
            results_file = os.path.join(f"{self.results_dir}/tasks", task_dir, "results.json")
            if not os.path.exists(results_file):
                continue

            with open(results_file, "r") as f:
                task_results = json.load(f)
            
            if task_key not in self.results:
                self.results[task_key] = []
            self.results[task_key] = task_results

        with open(self.results_dir + f"all_results.json", "w") as f:
            json.dump(self.results, f, indent=4)

    @profiler.profile
    def run_task(self, task, thread_id, thread_logger):
        task.assigned_llms = {}
        for model_id in task.models:
            if model_id is None:
                continue

            if self.resource_manager.is_api_model(model_id):
                thread_logger.info(f"Running task with API model {model_id}")
                llm = self.resource_manager.get_inactive_llm_instance(model_id)
                if llm:
                    task.assigned_llms[model_id] = llm
                    continue
                else:
                    thread_logger.info(f"API instance not found for model {model_id}. Skipping task.")
                    return False
            
            max_retries = 3
            for _ in range(max_retries):
                llm = self.resource_manager.get_inactive_llm_instance(model_id)
                if llm:
                    task.assigned_llms[model_id] = llm
                    break
                else:
                    thread_logger.info(f"LLM instance not found for model {model_id}. Retrying...")

                time.sleep(1)  # Wait before retrying

            if not task.assigned_llms[model_id]:
                thread_logger.info(f"Failed to assign GPU for model {model_id} after {max_retries} attempts. Skipping task.")
                return False

        if task.assigned_llms:
            thread_logger.info(f"Running task with LLMs: {[(llm.model_id, llm.gpu_ids) for llm in task.assigned_llms.values()]}")

        success = False
        try:
            start_time = time.time()

            ### RUN TASK ###
            self.experiment.run_task(
                thread_id, 
                task, 
                self.timestamp, 
                N=self.N, 
                logger=thread_logger
            )
            ### RUN TASK ###

            end_time = time.time()
            thread_logger.info(f"Matchup completed in {end_time - start_time:.2f} seconds")
            success = True
        except Exception as e:
            thread_logger.error(f"Error in task: {task.description} --> {e}")
            thread_logger.error(traceback.format_exc())

        for model_id in task.models:
            if not model_id is None and model_id in task.assigned_llms:
                self.resource_manager.release_llm(model_id, task.assigned_llms[model_id])

        if success:
            thread_logger.info(f"Finished task: {task.description}")

        return True

    def sample_task(self, tasks):
        
        if len(self.experiment.current_tasks) == 0:
            return random.choice(tasks)
        
        inactive_models = set(self.resource_manager.get_available_models())

        for task in tasks:
            if all(model_id in inactive_models for model_id in task.models):
                return task
            
        return random.choice(tasks)
    
    def run(self):
        self.initialize_gpus()
        self.logger.info("Initial state after GPU assignment:")
        self.print_state(iteration=0)

        threads = []
        for thread_id in range(self.max_concurrent_threads):
            t = threading.Thread(target=self.thread_worker, args=(thread_id,))
            t.start()
            threads.append(t)

        iteration = 0
        while self.experiment.tasks or not self.task_queue.empty():
            iteration += 1
            
            available_models = self.resource_manager.get_available_models()
            active_models = self.resource_manager.get_active_models()
            available_active_models = [m for m in available_models if m in active_models]

            if not available_active_models:
                info = "No available active models. Continuing to next iteration."
                self.print_state(iteration=iteration, info=info)
                time.sleep(self.wait_time)
                continue
            
            tasks = self.experiment.get_tasks_by_models(available_active_models)
            if not tasks:
                new_model = self.resource_manager.replace_models(available_active_models)
                warning = f"No valid tasks --> replaced {available_active_models} with {new_model}"
                self.print_state(iteration=iteration, warning=warning)
                time.sleep(self.wait_time)
                continue

            task = self.sample_task(tasks)
            
            if self.active_threads._value > 0:  # Check if there are available threads
                self.task_queue.put(task)
                self.experiment.remove_task(task)
                self.logger.info(f"  Task added to queue: {task}")
            else:
                self.logger.info("  Max concurrent threads reached. Waiting.")
                time.sleep(self.wait_time)

            self.update_results()
            self.print_state(iteration=iteration)

        for _ in range(self.max_concurrent_threads):
            self.task_queue.put(None)

        for t in threads:
            t.join()

        self.logger.info("\nAll tasks completed.")

    @profiler.profile
    def initialize_gpus(self):
        self.logger.info("Initializing GPU assignments:")
        models = list(self.resource_manager.llm_instances.keys())
        
        models.sort(key=lambda m: self.resource_manager.get_model_size(m), reverse=True)
        available_gpus = self.resource_manager.get_available_gpus()

        if not available_gpus:
            self.logger.info("  No available GPUs for initialization.")
            for model_id in models:
                if self.resource_manager.is_api_model(model_id):
                    self.logger.info(f"  Assigning API model: {model_id}")
                    self.resource_manager.assign_gpu(model_id)
                    continue

        count = 0
        while self.resource_manager.get_available_gpus():
            count += 1
            if count > 10:
                self.logger.info("  Failed to fill all GPUs, exiting initialization.")
                break
            for model_id in models:
                model_size = self.resource_manager.get_model_size(model_id)
                required_gpus = self.resource_manager.get_required_gpus(model_size)
                available_gpus = self.resource_manager.get_available_gpus()
                
                if len(available_gpus) >= required_gpus:
                    if self.resource_manager.assign_gpu(model_id):
                        assigned_gpus = self.resource_manager.llm_instances[model_id][-1]['gpus']
                        self.logger.info(f"  Assigned {assigned_gpus} to {model_id} ({model_size}B)")
                    else:
                        self.logger.info(f"  Failed to assign GPUs to {model_id} ({model_size}B)")
                else:
                    self.logger.info(f"  Not enough GPUs for {model_id} ({model_size}B). Required: {required_gpus}, Available: {len(available_gpus)}")

        # Log the final state after initialization
        self.logger.info("Final GPU assignments:")
        for model_id, instances in self.resource_manager.llm_instances.items():
            for idx, instance in enumerate(instances):
                self.logger.info(f"  {model_id} (Instance {idx}): GPUs {instance['gpus']}, Active: {instance['active']}")

        remaining_gpus = self.resource_manager.get_available_gpus()
        if remaining_gpus:
            self.logger.info(f"  Remaining unassigned GPUs: {remaining_gpus}")

    @profiler.profile
    def print_state(self, iteration: int, warning: str = None, info: str = None):
        """
        Print the current state of the experiment with iteration number, optional warning and info.
        
        Args:
            iteration: Current iteration number
            warning: Optional warning message to display prominently in red
            info: Optional info message to display prominently in green
        """
        # Display warning first if present
        if warning:
            self.print_console.print(Panel(
                f"[bold red]{warning}[/bold red]",
                border_style="red",
                title="[bold red]WARNING[/bold red]",
                padding=(1, 2)
            ))
            
        # Display info if present
        if info:
            self.print_console.print(Panel(
                f"[bold green]{info}[/bold green]",
                border_style="green",
                title="[bold green]INFO[/bold green]",
                padding=(1, 2)
            ))

        # Determine border style based on state
        border_style = "red" if warning else "green" if info else None
        title_color = "red" if warning else "green" if info else "magenta"

        # Primary stats table
        main_table = Table(
            title=f"[bold {title_color}]Iteration {iteration}",
            box=box.ROUNDED,
            border_style=border_style
        )
        main_table.add_column("Resource", style="cyan", justify="right")
        main_table.add_column("Status", style="green")
        main_table.add_column("Models", style="yellow", justify="right")
        main_table.add_column("Details", style="blue")

        # Combine general and resource information in one row
        main_table.add_row(
            f"Threads [{self.max_concurrent_threads - self.active_threads._value}/{self.max_concurrent_threads}]",
            f"Tasks [{len(self.experiment.tasks)}/{self.initial_num_tasks}]",
            f"GPUs [{len(self.resource_manager.get_available_gpus())}/{len(self.resource_manager.gpu_ids)}]",
            f"Queue Size [{self.task_queue.qsize()}]"
        )

        # Add active tasks in a compact format
        active_tasks = [f"{task}: {sum(progress):.3f}/{len(progress)}" 
                        for task, progress in self.experiment.current_tasks.items()]
        if active_tasks:
            main_table.add_row(
                "Active Tasks",
                "\n".join(active_tasks),
                "",
                f"Total: {len(self.experiment.current_tasks)}"
            )

        self.print_console.print(main_table)

        # Combine model states and profiling in one table
        stats_table = Table(
            title=f"Models & Performance (Iter {iteration})", 
            box=box.ROUNDED,
            border_style=border_style
        )
        stats_table.add_column("Model/Function", style="cyan")
        stats_table.add_column("Details", style="yellow")
        stats_table.add_column("Performance", style="magenta")

        # Add model information more concisely
        for model_id, instances in self.resource_manager.llm_instances.items():
            active_instances = sum(inst['active'] for inst in instances)
            gpu_info = [f"GPUs: {inst['gpus']}" if inst['gpus'] else "API" for inst in instances]
            stats_table.add_row(
                model_id,
                f"Size: {self.resource_manager.get_model_size(model_id)}B",
                f"Active: {active_instances}"
            )

        # Add a separator row
        stats_table.add_row("", "", "")

        # Add profiling information more concisely
        total_exec_time = time.time() - self.start_time
        stats_table.add_row(
            "Total Execution",
            f"{total_exec_time:.2f}s",
            f"Iter {iteration}"
        )

        # Only show functions with significant execution time (e.g., > 1% of total time)
        for func_name, times in self.profiler.function_times.items():
            if times:  # Only show if there are timestamps
                avg_time = sum(times) / len(times)
                total_time = sum(times)
                # Only show functions that took more than 1% of total time
                if total_time > total_exec_time * 0.01:
                    stats_table.add_row(
                        func_name,
                        f"Calls: {len(times)}",
                        f"Avg: {avg_time:.2f}s"
                    )

        self.print_console.print(stats_table)
        
        # Add status reminders at the bottom if present
        if warning:
            self.print_console.print("\n[bold red]⚠️  Warning state active ⚠️[/bold red]")
        if info:
            self.print_console.print("\n[bold green]ℹ️  Info state active ℹ️[/bold green]")
        
        self.print_console.print("\n" + "="*50 + "\n")