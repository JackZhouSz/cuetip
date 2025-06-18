import copy, time, random, math, os, threading, datetime, json

from tqdm import tqdm
from contextlib import contextmanager
from typing import List, Dict, Tuple

from poolagent.pool import *
from poolagent.path import ROOT_DIR

class TimeoutException(Exception):
    pass

class TimeoutLock(object):
    def __init__(self):
        self._lock = threading.Lock()

    def acquire(self, blocking=True, timeout=-1):
        return self._lock.acquire(blocking, timeout)

    @contextmanager
    def acquire_timeout(self, timeout):
        result = self._lock.acquire(timeout=timeout)
        yield result
        if result:
            self._lock.release()

    def release(self):
        self._lock.release()

def run_with_timeout(func, args=(), kwargs={}, timeout_duration=1):
    result = [TimeoutException("Function call timed out")]
    lock = TimeoutLock()

    def target():
        with lock.acquire_timeout(timeout_duration) as acquired:
            if acquired:
                try:
                    result[0] = func(*args, **kwargs)
                except Exception as e:
                    result[0] = e
            else:
                result[0] = TimeoutException("Function call timed out")

    thread = threading.Thread(target=target)
    thread.start()
    thread.join(timeout_duration + 0.1)  # Give a little extra time for the lock to be released
    return result[0]

class MCTSNode:
    def __init__(self, state: State, events: List[Event], player: str = "one", double_shot: bool = False, parent=None, action=None):
        self.state = state
        self.events = events
        self.parent = parent
        self.children: List[MCTSNode] = []
        self.visits = 0
        self.value = 0.0
        self.player = player
        self.double_shot = double_shot

        self.action = action

class MCTS:
    def __init__(self, game: PoolGame, propose_shot, iterations=250, exploration_weight=1.414, branching_factor=10):
        self.game = game
        self.propose_shot = propose_shot
        self.iterations = iterations
        self.exploration_weight = exploration_weight
        self.branching_factor = branching_factor
        self.timeout = 10
        self.total_sims = 0

    def select(self, node: MCTSNode) -> MCTSNode:
        while node.children:
            node = self.best_child(node)
        return node

    def expand(self, node: MCTSNode) -> MCTSNode:
        
        self.game.from_state(node.state)
        if self.game.check_win():
            return node
        
        if len(node.children) >= self.branching_factor:
            return node

        while len(node.children) < self.branching_factor:
            self.game.current_player = node.player
            self.game.double_shot = node.double_shot
            action = self.propose_shot(self.game)
            if self.apply_action(node.state, action):
                new_state = self.game.get_state()
                events = self.game.get_events()
                child_node = MCTSNode(new_state, events, player=self.game.current_player, double_shot=self.game.double_shot, parent=node, action=action)
                child_node.state.params = action
                node.children.append(child_node)

        return random.choice(node.children)

    def simulate(self, node: MCTSNode) -> float:
        state = copy.deepcopy(node.state)
        self.game.from_state(state)

        self.game.current_player = node.player

        while not self.game.check_win():
            action = self.propose_shot(self.game, eps=0.0)
            state = self.game.get_state()
            self.apply_action(state, action)

        reward = self.game.reward()
        self.game.current_player = node.player
        return reward[0] 

    def backpropagate(self, node: MCTSNode, result: float):
        while node is not None:
            node.visits += 1
            node.value += result 
            node = node.parent

    def best_child(self, node: MCTSNode) -> MCTSNode:
        return max(node.children, key=lambda c: self.uct_score(c))

    def uct_score(self, node: MCTSNode) -> float:
        exploitation = node.value / node.visits if node.visits > 0 else 0
        exploration = math.sqrt(math.log(node.parent.visits) / node.visits) if node.visits > 0 else float('inf')
        return exploitation + self.exploration_weight * exploration

    def apply_action(self, state: State, action: Dict) -> bool:
        self.game.from_state(state)
        self.total_sims += 1
        return self.game.take_shot(self.game.current_player, action)

    def run(self, root_state: State, save_path) -> Tuple[State, List[Tuple[State, int]]]:
        start_time = time.time()

        self.game.from_state(root_state)
        root = MCTSNode(root_state, [])
        self.expand(root)

        for idx in tqdm(range(self.iterations), desc="Performing MCTS"):
            node = self.select(root)
            node = self.expand(node)
            result = self.simulate(node)
            self.backpropagate(node, result)

            if (idx+1) % 100 == 0:
                processed_dataset = process_dataset([root], idx+1)
                with open(save_path, "w") as f:
                    json.dump(processed_dataset, f, indent=4)

        # Print stats
        run_time = time.time() - start_time
        print("--- stats ---")

        print(f"Total Time: {run_time:3f}s")
        print(f"Total Sims: {self.total_sims}")
        print(f"Average Sims: {self.total_sims/self.iterations:.3f}")
        print(f"Average Sim/s: {self.total_sims/run_time:3f}")

        print("-------------")

        return root

def run_multiple_mcts(states: List[State], game: PoolGame, propose_shot, iterations: int, branching_factor: int = 10) -> List[Tuple[State, List[Tuple[State, int]]]]:

    save_dir = ROOT_DIR + "/poolagent/value_data/logs/mcts"
    os.makedirs(save_dir, exist_ok=True)

    for idx, state in enumerate(states):
        print(f"--- State {idx+1}/{len(states)} ---")
        mcts = MCTS(game, propose_shot, iterations=iterations, branching_factor=branching_factor)

        time = datetime.datetime.now().strftime("%H:%M:%S")
        save_path = f"{save_dir}/mcts_data_{time}.json"

        run_data = mcts.run(state, save_path)
        game.reset()

        processed_dataset = process_dataset([run_data], iterations)
        with open(save_path, "w") as f:
            json.dump(processed_dataset, f, indent=4)

    return processed_dataset

def process_dataset(dataset: List[MCTSNode], iterations: int) -> List[Dict]:
    processed_data = []
    for node in dataset:
        
        # Skip nodes without children
        if not node.children:
            continue

        # Require a minimum number of visits to consider the node
        if node.visits / iterations < 0.15:
            continue

        total_visits = sum([child.visits for child in node.children])
        distribution = [child.visits / total_visits for child in node.children]
        
        processed_data.append({
            "player": node.player,
            "starting_state": node.state.to_json(),
            "follow_up_states": [child.state.to_json() for child in node.children],
            "events": [[[event.encoding, event.pos] for event in child.events] for child in node.children],
            "visit_distribution": distribution
        })

        child_data = [process_dataset(child.children, iterations) for child in node.children]
        for data in child_data:
            processed_data.extend(data)

    return processed_data 
