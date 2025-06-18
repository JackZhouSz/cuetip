import random, PIL, io, base64, os
import numpy as np
import pooltool as pt

from typing import Dict, List

from typing import List
from pooltool.ani.image import get_graphics_texture, image_array_from_texture
from pooltool.ani.camera import cam, camera_states

from poolagent.utils import State, Event, Image, get_image_util

LIMITS = {
    "V0": (0.25, 4),   # CUE BALL SPEED
    "phi": (0, 360),  # CUE ANGLE
    "theta": (5, 60), # CUE INCLINATION
    "a": (-0.25, 0.25),      # CUE OFFSET
    "b": (-0.25, 0.25)       # CUE OFFSET
}

STANDARD_IDS = ['red', 'blue', 'yellow', 'brown', 'black', 'green', 'pink']
SNOOKER_IDS = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15']

CONVERT_BALLS_FOR_VISUAL = {
    "black": "7",
    "pink": "8",
    "green": "6",
}

class Fouls:
    """The fouls that can be made in a pool game

    Returns:
        _type_: int -- The type of foul made
    """
    NONE              = 0
    POT_CUE_BALL      = 1
    POT_OPP_BALL      = 2
    NO_CONTACT        = 3
    CONTACT_OPP_FIRST = 4

    def string(foul : int) -> str:
        """Returns a string representation of the foul

        Args:
            foul (int): The foul that was made

        Returns:
            str: The string representation of the foul
        """

        if foul not in [0,1,2,3,4]:
            return
        
        if foul == 0:
            return "No foul"
        
        elif foul == 1:
            return "The cue ball was potted"
        
        elif foul == 2:
            return "One of the opponents balls where potted"
        
        elif foul == 3:
            return "No contact with one of your balls was made"
        
        elif foul == 4:
            return "Contact was made with one of the opponents balls first"

class PoolMaster:
    def calculate_indirect_shot_difficulty(state: State, ball: str, pocket: str) -> float:
        
        # Invert table (y-axis) and move cue ball to x + 1
        new_positions = {
            k: [1 - p[0], p[1]] for k, p in state.ball_positions.items()
        }
        cue_pos = state.ball_positions["cue"]
        cue_pos = [cue_pos[0] + 1, cue_pos[1]]
        temp_state = State(positions={**new_positions, "cue": cue_pos})

        # Get new pocket as its inverted 
        if 'l' in pocket:
            new_pocket = pocket.replace('l','r')
        elif 'r' in pocket:
            new_pocket = pocket.replace('r','l')

        # Calculate shot same as direct shot 
        return PoolMaster.evaluate_shot(temp_state, ball, new_pocket, no_indirect=True)

    def evaluate_shot(state: State, ball: str, pocket: str, no_indirect=False) -> float:
        """Calculate the difficulty coefficient (κ) for a given shot.

        For corner pockets: κc = cos(α) / (v_op * v_co)
        For side pockets: κs = cos(α) * cos(γ) / (v_op * v_co)
        where:
        - α is the cut angle between object-pocket and cue-object vectors
        - γ is the angle between object-pocket vector and x-axis (for side pockets)
        - v_op is the distance between object ball and pocket
        - v_co is the distance between cue ball and object ball
        """

        MIN_DISTANCE = 0.1

        # Check if all required positions are available
        if ball not in state.ball_positions or pocket not in state.pocket_positions:
            return float('inf')

        ball_pos = state.get_ball_position(ball)
        pocket_pos = state.get_pocket_position(pocket)
        cue_pos = state.get_ball_position("cue")

        # Check for invalid positions
        if np.isinf(ball_pos).any() or np.isinf(pocket_pos).any() or np.isinf(cue_pos).any():
            return float('inf')

        if not no_indirect:
            if not state.line_of_sight(ball_pos, pocket_pos) or not state.line_of_sight(ball_pos, cue_pos):
                return PoolMaster.calculate_indirect_shot_difficulty(state, ball, pocket)

        # Calculate vectors
        v_op = np.array([pocket_pos[0] - ball_pos[0], pocket_pos[1] - ball_pos[1]])
        v_co = np.array([ball_pos[0] - cue_pos[0], ball_pos[1] - cue_pos[1]])

        # Calculate vector magnitudes
        v_op_norm = max(np.linalg.norm(v_op), MIN_DISTANCE)
        v_co_norm = max(np.linalg.norm(v_co), MIN_DISTANCE)

        if v_op_norm == 0 or v_co_norm == 0:
            return float('inf')

        # Normalize vectors
        v_op_hat = v_op / v_op_norm
        v_co_hat = v_co / v_co_norm

        # Calculate cut angle cosine
        cos_alpha = np.dot(v_op_hat, v_co_hat)

        # if not no_indirect:
        #     # If angle too sharp for direct shot
        #     if cos_alpha < 0.1:
        #         return PoolMaster.calculate_indirect_shot_difficulty(state, ball, pocket)

        # Base difficulty coefficient
        difficulty = cos_alpha / (v_op_norm * v_co_norm)

        # For side pockets (lc = left center, rc = right center)
        if pocket in ['lc', 'rc']:
            # Calculate angle with x-axis
            x_axis = np.array([1.0, 0.0])
            cos_gamma = np.dot(v_op_hat, x_axis)
            
            # Check cut-off value for impossible angles
            if abs(cos_gamma) < 0.25:
                return float('inf')
            
            # Adjust difficulty for side pocket
            difficulty *= cos_gamma

        return 1.0 - difficulty

    def evaluate_position(state: State, pos: np.ndarray, target_balls: List[str]) -> float:
        """Evaluate a position's overall quality using average quality coefficient."""
        if not target_balls:
            return float('inf')
        
        temp_state = State(positions={**state.ball_positions, "cue": pos.tolist()})
        qualities = []

        if temp_state.balls_overlapping():
            return float('inf')
        
        for ball in target_balls:
            if ball not in temp_state.ball_positions:
                continue
                
            ball_qs = []
            for pocket in state.pocket_positions.keys():
                # Check line of sight
                ball_pos = temp_state.get_ball_position(ball)
                if np.isinf(ball_pos).any():
                    continue
                    
                if temp_state.line_of_sight(pos, ball_pos) and temp_state.line_of_sight(ball_pos, temp_state.get_pocket_position(pocket)):
                    quality = PoolMaster.evaluate_shot(temp_state, ball, pocket)
                    if quality != float('inf'):
                        ball_qs.append(quality)
            
            if ball_qs:
                qualities.append(min(ball_qs))
        
        if not qualities:
            return float('inf')
        
        return np.mean(qualities)

    def find_best_position(state: State, initial_positions: List[np.ndarray], target_balls: List[str]) -> np.ndarray:
        """Find the best position using gradient descent from multiple starting points."""
        best_position = None
        best_score = float('inf')
        
        for init_pos in initial_positions:
            current_pos = init_pos.copy()
            current_score = PoolMaster.evaluate_position(state, current_pos, target_balls)
            step_size = state.ball_radius * 2
            
            for _ in range(35):  # Maximum iterations
                gradients = []
                for dx, dy in [(step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)]:
                    test_pos = current_pos + np.array([dx, dy])
                    if 0 <= test_pos[0] <= state.table_width_height[0] and 0 <= test_pos[1] <= state.table_width_height[1]:
                        score = PoolMaster.evaluate_position(state, test_pos, target_balls)
                        gradients.append((score, np.array([dx, dy])))
                
                if not gradients:
                    break
                    
                best_grad = min(gradients, key=lambda x: x[0])
                if best_grad[0] >= current_score:
                    break
                    
                current_pos += best_grad[1]
                current_score = best_grad[0]
                
                if current_score < best_score:
                    best_score = current_score
                    best_position = current_pos.copy()
        
        return best_position if best_position is not None else initial_positions[0]

    def visualize_shot_analysis(state: State, 
                            final_state: State, 
                            events: List[Event], 
                            grid_positions: List[np.ndarray], 
                            grid_values: List[float],
                            target_balls: List[str],
                            save_path: str = None) -> None:
        """
        Creates a visualization showing two heatmaps:
        1. Initial state position quality
        2. Final state position quality
        """
        import matplotlib.pyplot as plt
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
        
        # Function to create heatmap data
        def create_heatmap_data(values):
            Z = np.full((len(y_unique), len(x_unique)), np.nan)    

            for pos, val in zip(grid_positions, values):
                x_idx = np.where(x_unique == pos[0])[0][0]
                y_idx = np.where(y_unique == pos[1])[0][0]
                Z[y_idx, x_idx] = val
                
            return Z

        # Convert grid positions to unique coordinates
        x_coords = [pos[0] for pos in grid_positions]
        y_coords = [pos[1] for pos in grid_positions]
        x_unique = np.unique(x_coords)
        y_unique = np.unique(y_coords)

        # Invert grid_values

        for i in range(len(grid_values)):
            if grid_values[i] == float('inf'):
                grid_values[i] = 0

        max_val = max(grid_values)
        grid_values = [ val / max_val for val in grid_values]
        # grid_values = [ np.exp(-val) for val in grid_values]
        
        grid_values = create_heatmap_data(grid_values)
        
        # Plot both heatmaps
        for ax, Z, title, current_state in [(ax1, grid_values, 'Initial State', state),
                                        (ax2, grid_values, 'Final State', final_state)]:
            # Plot heatmap
            im = ax.imshow(Z, 
                        extent=[
                            0, 
                            state.table_width_height[0], 
                            0, 
                            state.table_width_height[1]
                        ],
                        origin='lower', 
                        cmap='summer', 
                        aspect='auto')
            
            # Plot balls
            for ball, pos in current_state.ball_positions.items():
                if not np.isinf(pos).any() and not isinstance(pos[0], str):
                    color = 'white' if ball == 'cue' else ball
                    ax.plot(pos[0], pos[1], 'o', color=color,
                        markersize=20, markeredgecolor='black', linewidth=2)
            
            # Plot pockets
            for pocket, pos in state.pocket_positions.items():
                ax.plot(pos[0], pos[1], 's', color='black', markersize=10)
            
            # Add table boundary
            ax.plot([0, state.table_width_height[0], state.table_width_height[0], 0, 0],
                    [0, 0, state.table_width_height[1], state.table_width_height[1], 0],
                    'k-', linewidth=2)
            
            ax.set_title(title)
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Position Quality')
        
        plt.suptitle('Shot Analysis - Position Quality Comparison', y=1.02)
        
        if save_path:
            plt.savefig(save_path, 
                        bbox_inches='tight',
                        dpi=300,
                        pad_inches=0.2)
        else:
            plt.tight_layout()
            plt.show()
        
        plt.close()

class Pool():
    """Pool environment that attempts to follow the rules of pool but alerting the shot taker if a foul was made.
    Fouls:
        - Pot cue ball
        - Potting an opponents ball
        - Dont contact any of your balls 
        - Contact a ball of the opponents first
    """

    STEPPER = pt.FrameStepper()

    def __init__(self, mode="threeball", visualizable=False) -> None:
        """Initialise the pool environment

        Args:
            mode (str, optional): The mode of the pool game, options: ['threeball', 'sixball']. Defaults to "threeball".
            visualizable (bool, optional): If the pool game should be visualizable. Defaults to False.
        """

        self.visualizable = visualizable
        self.interface = None
        
        if self.visualizable:
            self.interface = pt.ShotViewer()

        self.mode = mode
        self.table = pt.Table.default()
        self.balls = self.setup_balls(self.mode)
        self.cue = pt.Cue(cue_ball_id="cue")
        self.shot = pt.System(table=self.table, balls=self.balls, cue=self.cue)

    def setup_balls(self, mode) -> dict:
        """Setup the balls on the table, based on the mode of the game, options: ['threeball', 'sixball']

        Args:
            mode (_type_): The mode of the game

        Returns:
            dict: The balls on the table
        """

        if mode == "threeball":

            x1, y1 = 0.25, 1.5
            x2, y2 = 0.5, 1.5
            x3, y3 = 0.75, 1.5

            return {
                "cue": pt.Ball.create("cue", xy=(0.5, 0.5)),
                "red": pt.Ball.create("red", xy=(x1, y1), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "yellow": pt.Ball.create("yellow", xy=(x2, y2), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "blue": pt.Ball.create("blue", xy=(x3, y3), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            }
        
        if mode == "sixball":

            p1_x1, p1_y1 = 0.25, 1.75
            p1_x2, p1_y2 = 0.25, 1.5
            p1_x3, p1_y3 = 0.25, 1.25

            p2_x1, p2_y1 = 0.75, 1.75
            p2_x2, p2_y2 = 0.75, 1.5
            p2_x3, p2_y3 = 0.75, 1.25

            return {
                "cue": pt.Ball.create("cue", xy=(0.5, 0.5)),

                "red": pt.Ball.create("red", xy=(p1_x1, p1_y1), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "yellow": pt.Ball.create("yellow", xy=(p1_x2, p1_y2), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "blue": pt.Ball.create("blue", xy=(p1_x3, p1_y3), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),

                "green": pt.Ball.create("green", xy=(p2_x1, p2_y1), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "black": pt.Ball.create("black", xy=(p2_x2, p2_y2), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
                "pink": pt.Ball.create("pink", xy=(p2_x3, p2_y3), ballset=pt.game.layouts.DEFAULT_SNOOKER_BALLSET),
            }

    def reset(self) -> None:
        """Reset the pool environment to its initial state
        """

        self.shot.reset_history()
        self.shot.reset_balls()

        self.table = pt.Table.default()
        self.balls = self.setup_balls(self.mode)
        self.cue = pt.Cue(cue_ball_id="cue")

        del self.shot.table 
        del self.shot.balls
        del self.shot.cue

        self.shot.table = self.table
        self.shot.balls = self.balls
        self.shot.cue = self.cue

    def from_state(self, state : State, reset=True) -> bool:
        """Update the PoolTool state to the provided state

        Args:
            state (State): The state to update the pool environment to
            reset (bool, optional): If the pool environment should be reset before updating.

        Raises:
            Exception: If the state is not in the correct format

        Returns:
            bool: If the state was successfully updated
        """

        if isinstance(state, dict):
            try:
                state = State.from_json(state)
            except:
                raise Exception("Invalid state format.")

        self.balls = {}

        positions = state.ball_positions

        for ball, x_y in positions.items():
            x, y = x_y

            if x == np.inf or y == np.inf or isinstance(x, str) or isinstance(y, str):
                continue

            ballset = pt.game.layouts.DEFAULT_SNOOKER_BALLSET if ball in STANDARD_IDS else pt.game.layouts.DEFAULT_STANDARD_BALLSET

            if ball == "cue":
                self.balls[ball] = pt.Ball.create(ball, xy=(x, y))
            else:
                self.balls[ball] = pt.Ball.create(ball, xy=(x, y), ballset=ballset)

        self.shot.balls = self.balls
        
        return True

    def check_rules(self, target_balls) -> int:
        """Check if the current shot breaks any of the pool rules. This is done by checking the events that occured during the shot.

        Args:
            target_balls (_type_): The balls that the player is aiming to pot (this is very important for the rules)

        Returns:
            int: The foul that was made
        """

        events = self.get_events()
        opp_balls = [ball for ball in self.balls.keys() if ball not in target_balls]
        opp_balls.remove("cue")

        # 1. Potting cue ball
        if any(['ball-pocket-cue' in event.encoding for event in events]):
            return Fouls.POT_CUE_BALL
        
        # 2. Pot opponents ball
        if any([
            Event.ball_pocket(ball) in events for ball in opp_balls
        ]):
            return Fouls.POT_OPP_BALL
        
        # 4. Contact ball of opponents first
        for event in events:
            if event.encoding.startswith("ball-ball"):
                if any([ball in event.encoding for ball in opp_balls]):
                    return Fouls.CONTACT_OPP_FIRST
                else:
                    break

        # 3. Don't contact any of your balls 
        if not any([
            Event.ball_collision("cue", ball) in events for ball in target_balls
        ]):
            return Fouls.NO_CONTACT
            
        return Fouls.NONE

    def strike(self, V0 : float, phi : float, theta : float, a : float, b : float, check_rules=False, target_balls=[]) -> int:
        """Take a shot in the pool game and simulate the outcome

        Args:
            V0 (float): The power of the shot
            phi (float): The incline of the cue
            theta (float): The angle of the cue
            a (float): The x offset of the cue on the ball
            b (float): The y offset of the cue on the ball
            check_rules (bool, optional): If the rules should be checked. Defaults to False.
            target_balls (list, optional): The balls that the player is aiming to pot. Defaults to [].

        Returns:
            bool: If the shot was successful
        """
        self.shot.strike(V0=V0, phi=phi, theta=theta, a=a, b=b)
        pt.simulate(self.shot, inplace=True)
        if check_rules and target_balls:
            return self.check_rules(target_balls)
        else:
            return Fouls.NONE
        
    def random_params(self) -> dict:
        """Generate random parameters for the shot

        Returns:
            dict: The random parameters, within the limits
        """
        return {
            "V0": random.uniform(LIMITS["V0"][0], LIMITS["V0"][1]),
            "theta": random.uniform(LIMITS["theta"][0], LIMITS["theta"][1]),
            "phi": random.uniform(LIMITS["phi"][0], LIMITS["phi"][1]),
            "a": random.uniform(LIMITS["a"][0], LIMITS["a"][1]),
            "b": random.uniform(LIMITS["b"][0], LIMITS["b"][1])
        }

    def random_strike(self, check_rules=False, target_balls=[]) -> dict:
        """Take a random strike in the pool game

        Args:
            check_rules (bool, optional): If the rules should be checked. Defaults to False.
            target_balls (list, optional): The balls that the player is aiming to pot. Defaults to [].

        Returns:
            dict: The parameters of the random strike
        """
        params = self.random_params()
        if self.strike(**params, check_rules=check_rules, target_balls=target_balls) == Fouls.NONE:
            return params
        else:
            # TODO: Handle random strike case
            return {}

    def get_state(self) -> State:
        """Get the current state of the pool environment

        Returns:
            State: The current state of the pool environment
        """
        board_state = self.shot.get_board_state()
        state = State()
        state.from_board_state(board_state)
        return state

    def visualise(self) -> None:
        """Visualise the pool environment using PoolTool's visualisation tool

        Raises:
            Exception: If the pool environment is not visualizable
        """
        if self.visualizable:
            self.interface.show(self.shot)
            self.interface.stop()
        else:
            raise Exception("Cannot visualize without visualizable=True")
    
    def get_events(self) -> List[Event]:
        """Get the events that occurred during the last shot as a list of event objects.

        Returns a list of events starting from the latest shot:
            - collision between balls, cushion, cue stick ;
            - state transition of balls from rolling to sliding etc...
            - pocketing events, bool id & pocket id.

        Returns:
            List[Event]: The events that occurred during the last shot
        """

        pt_events = self.shot.events
        events = []

        for e in pt_events:
            typ_obj : tuple[str, str, str] = e.typ_obj

            if typ_obj == ("","",""):
                continue

            encoding = typ_obj[0] + "-" + typ_obj[1] 
            pos = typ_obj[2]

            events.append(Event.from_encoding(encoding, pos))

        events = [e for e in events if len(e.encoding) > 0]

        return events
    
    def get_image(self, camera_state_name: str = "7_foot_overhead", resolution=(1920, 1080)) -> Image:
        """Use the pooltool image API to return an image of the current state

        Args:
            camera_state_name (str, optional): Name of the camera state to use. Defaults to "7_foot_overhead".
            resolution (tuple, optional): Resolution of the output image as (width, height). Defaults to (1920, 1080).

        Returns:
            Image: The image of the current state
        """

        self.balls = {}

        positions = self.get_state().ball_positions

        for ball, x_y in positions.items():
            x, y = x_y

            if x == np.inf or y == np.inf or isinstance(x, str) or isinstance(y, str):
                continue

            if ball == "cue":
                self.balls[ball] = pt.Ball.create(ball, xy=(x, y))
            else:

                ballset = pt.game.layouts.DEFAULT_SNOOKER_BALLSET if ball in STANDARD_IDS else pt.game.layouts.DEFAULT_STANDARD_BALLSET

                self.balls[ball] = pt.Ball.create(ball, xy=(x, y), ballset=ballset)
                self.balls[ball].initial_orientation = pt.BallOrientation(
                    pos=(1.0, 1.0, 1.0, 1.0),
                    sphere=(1, 0, 0, 0),
                )

        self.shot.balls = self.balls
        
        if 'cue' not in self.shot.balls:
            self.shot.balls['cue'] = pt.Ball.create('cue', xy=(float('inf'), float('inf')))

        if Pool.STEPPER is None:
            return get_image_util(self.get_state())

        pt.simulate(self.shot, inplace=True)

        if camera_state_name not in camera_states:
            raise ValueError(f"Invalid camera state name: {camera_state_name}. Available states: {list(camera_states.keys())}")

        camera_state = camera_states[camera_state_name]
        size=(resolution[0], resolution[1]//1.6)
        fps = 1

        iterator, frames = Pool.STEPPER.iterator(self.shot, size, fps)
        tex = get_graphics_texture()
        cam.load_state(camera_state)

        # Initialize a numpy array image stack
        imgs: List[np.NDArray[np.uint8]] = []

        for frame in range(frames):
            next(iterator)
            imgs.append(image_array_from_texture(tex, gray=False))

        image = np.array(imgs, dtype=np.uint8)[0]
        pil_image = PIL.Image.fromarray(image).rotate(90, expand=True)
        buffered = io.BytesIO()
        pil_image.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return Image(
            img_str,
            image
        )
    
    def get_balls(self) -> str:
        """Return a sring of the current balls on the table, not including those that were potted.

        Returns:
            str: The current balls on the table
        """

        current_balls = []

        balls = self.shot.get_board_state()["balls"]

        for key, el in balls.items():
            current_balls.append(key)

        return ", ".join(current_balls)
    
    def save_shot_gif(self, state : State, params : dict, path : str):
        """Use the pooltool stepper to save a gif of a single shot

        Args:
            state (State): State of the pool environment
            params (dict): Parameters of the shot
            path (str): Path to save the gif
        """
        
        if Pool.STEPPER is None:
            print("Pool.STEPPER is None - cannot save gif")
            return None
              
        self.balls = {}

        positions = state.ball_positions

        for ball, x_y in positions.items():
            x, y = x_y

            if x == np.inf or y == np.inf or isinstance(x, str) or isinstance(y, str):
                continue

            if ball == "cue":
                self.balls[ball] = pt.Ball.create(ball, xy=(x, y))
            else:

                ballset = pt.game.layouts.DEFAULT_SNOOKER_BALLSET if ball in STANDARD_IDS else pt.game.layouts.DEFAULT_STANDARD_BALLSET
                if ball in CONVERT_BALLS_FOR_VISUAL:
                    ball = CONVERT_BALLS_FOR_VISUAL[ball]
                    ballset = pt.game.layouts.DEFAULT_STANDARD_BALLSET

                self.balls[ball] = pt.Ball.create(ball, xy=(x, y), ballset=ballset)
                self.balls[ball].initial_orientation = pt.BallOrientation.random()

        self.shot.balls = self.balls
        
        self.strike(**params)

        camera_state = camera_states["7_foot_overhead"]
        size=(512, 512/1.6)
        fps = 30

        iterator, frames = Pool.STEPPER.iterator(self.shot, size, fps)
        tex = get_graphics_texture()
        cam.load_state(camera_state)

        # Initialize a numpy array image stack
        imgs: List[np.NDArray[np.uint8]] = []

        for frame in range(frames):
            next(iterator)
            imgs.append(image_array_from_texture(tex, gray=False))

        images = np.array(imgs, dtype=np.uint8)
        pil_images = (PIL.Image.fromarray(image).rotate(90, expand=True) for image in images)
        img = next(pil_images)
        img.save(
            fp=path,
            format="GIF",
            append_images=pil_images,
            save_all=True,
            duration=(1 / fps) * 1e3,
            loop=0,  # loop infinitely
        )

    def attempt_aim_at_ball(self, target_ball) -> float:
        """Attempt to find a phi value that aims at a particular ball, returns -1 if fails

        Args:
            target_ball (_type_): The ball to aim at

        Returns:
            float: The phi value of the shot
        """

        try:
            self.shot.aim_at_ball(target_ball)
            return self.shot.cue.phi
        except:
            return -1
    
    def attempt_pot_ball(self, target_ball) -> float:
        """Attempt to find a phi value that pots a particular ball in the best pocket, returns -1 if fails

        Args:
            target_ball (_type_): The ball to pot

        Returns:
            float: The phi value of the shot
        """
        try:
            self.shot.aim_for_best_pocket(target_ball)
            return self.shot.cue.phi
        except:
            return -1
        
    def attempt_pot_ball_pocket(self, target_ball, target_pocket) -> float:
        """Attempt to find a phi value that pots a particular ball in a particular pocket, returns -1 if fails

        Args:
            target_ball (_type_): The ball to pot
            target_pocket (_type_): The pocket to pot the ball in

        Returns:
            float: The phi value of the shot
        """

        try:
            self.shot.aim_for_pocket(target_ball, target_pocket)
            return self.shot.cue.phi
        except:
            return -1

    def calculate_pool_master_shot(self, state: State, target_balls: List[str]) -> Dict:
        """Calculate the optimal shot parameters based on the PoolMaster algorithm."""
        # If no balls to target, return empty parameters
        if not target_balls:
            return self.random_params()
                        
        # Find best shot and pocket combination for each ball
        #print("Finding best shots...")
        best_shots = []
        for ball in target_balls:
            if ball not in state.ball_positions:
                continue
                
            ball_pos = state.get_ball_position(ball)
            if np.isinf(ball_pos).any():
                continue
                
            for pocket in state.pocket_positions.keys():
                difficulty = PoolMaster.evaluate_shot(state, ball, pocket)
                if difficulty != float('inf'):
                    best_shots.append((ball, pocket, difficulty))
        
        if not best_shots:
            print("No possible shots found.")
            return self.random_params()
        
        # Sort shots by difficulty
        best_shots.sort(key=lambda x: x[2])
        
        # Try shots starting from the easiest
        for target_ball, pocket, _ in best_shots:
            #print(f"Calculating shot for {target_ball} to {pocket}...")

            # Set up initial shot parameters
            self.from_state(state)
            phi = self.attempt_pot_ball_pocket(target_ball, pocket)
            
            if phi == -1:
                continue
            
            # Generate grid of positions for repositioning
            #print("Calculating grid positions...")
            grid_positions = []
            for x in np.linspace(state.ball_radius, state.table_width_height[0] - state.ball_radius, 12):
                for y in np.linspace(state.ball_radius, state.table_width_height[1] - state.ball_radius, 12):
                    grid_positions.append(np.array([x, y]))
            
            # Find best position for next shot
            remaining_balls = [b for b in target_balls if b != target_ball and b in state.ball_positions]
            best_position = PoolMaster.find_best_position(state, grid_positions, remaining_balls)
            
            # Try different shot parameters to achieve position
            best_params = None
            best_position_error = float('inf')
            
            # Calculate grid position values once
            # print("Calculating grid values...")
            # grid_values = []
            # for pos in grid_positions:
            #     temp_state = State(positions={**state.ball_positions, "cue": pos.tolist()})
            #     qualities = []
            #     for ball in remaining_balls:
            #         if ball not in temp_state.ball_positions:
            #             continue
            #         ball_qs = []
            #         ball_pos = temp_state.get_ball_position(ball)
            #         if np.isinf(ball_pos).any():
            #             continue
            #         for pocket in state.pocket_positions:
            #             pocket_pos = temp_state.get_pocket_position(pocket)
            #             if (temp_state.line_of_sight(pos, ball_pos) and 
            #                 temp_state.line_of_sight(ball_pos, pocket_pos)):
            #                 q = PoolMaster.evaluate_shot(temp_state, ball, pocket)
            #                 if q != float('inf'):
            #                     ball_qs.append(q)
            #         if ball_qs:
            #             qualities.append(max(ball_qs))
            #     grid_values.append(np.mean(qualities) if qualities else float('inf'))
            
            #print("Calculating best shot parameters...")
            iteration = 0
            for V0 in np.linspace(0.5, 2.5, 8):
                for a in np.linspace(-0.15, 0.15, 5):
                    for b in np.linspace(-0.15, 0.15, 5):
                        params = {
                            "V0": V0,
                            "phi": phi,
                            "theta": 14,  
                            "a": a,
                            "b": b
                        }
                        
                        # Test shot
                        self.from_state(state)
                        foul = self.strike(**params, check_rules=True, target_balls=target_balls)
                        
                        if foul == 0:  # No foul
                            final_state = self.get_state()
                            if "cue" in final_state.ball_positions and target_ball not in final_state.ball_positions:
                                cue_final_pos = final_state.get_ball_position("cue")
                                if not np.isinf(cue_final_pos).any():
                                    position_error = np.linalg.norm(cue_final_pos - best_position)
                                    
                                    # Visualize every 50th iteration or when we find a better shot
                                    # if position_error < best_position_error:
                                    #     PoolMaster.visualize_shot_analysis(
                                    #         state=state,
                                    #         final_state=final_state,
                                    #         events=self.get_events(),
                                    #         grid_positions=grid_positions,
                                    #         grid_values=grid_values,
                                    #         target_balls=target_balls,
                                    #         save_path=f'shot_analysis_{iteration}.png'
                                    #     )
                                    
                                    if position_error < best_position_error:
                                        best_position_error = position_error
                                        best_params = params
                                        
                        iteration += 1
            
            return best_params if best_params is not None else self.random_params()
        
        print("No possible shots found.")

        return self.random_params()


class PoolGame(Pool):
    """This is a special version of the Pool class that is used for the game environment. It has additional methods for checking if a player has won and for rewarding the players. It also keeps track of the current player and the target balls of each player.
    """

    def __init__(self, visualizable=False, target_balls=None) -> None:
        """Initialise the PoolGame environment

        Args:
            visualizable (bool, optional): If the pool game should be visualizable. Defaults to False.
        """
        super().__init__("sixball", visualizable)

        self.players = ["one", "two"]
        self.current_player = "one"
        self.double_shot = False

        self.target_balls = {
            "one": ["red", "blue", "yellow"],
            "two": ["green", "black", "pink"]
        } if target_balls is None else target_balls

    def reset(self) -> None:
        """Reset the pool game environment, also reset the current player and double shot
        """
        super().reset()

        self.current_player = "one"
        self.double_shot = False

    def reset_cue_ball(self):
        """Reset the cue ball to a default position after it was potted
        """

        state = self.get_state()

        state.ball_positions["cue"] = [0.5, 0.5]

        def random_point_in_cue_ball_area():
            x = random.uniform(0.05, 0.95)
            y = random.uniform(0.05, 0.45)
            return [x, y]

        recursion_count = 0
        recursion_max = 250
        while state.balls_overlapping():
            recursion_count += 1
            if recursion_count > recursion_max:
                raise Exception("Could not find a non-overlapping position for the cue ball")
            state.ball_positions["cue"] = random_point_in_cue_ball_area()

        self.from_state(state, reset=False)

    def take_shot(self, player, params) -> bool:
        """Take a shot in the pool game, check if the shot was successful and if a foul was made. If a foul was made, change the current player and reset the cue ball if it was potted.

        Args:
            player (_type_): The player taking the shot
            params (_type_): The parameters of the shot

        Returns:
            bool: If the shot was a foul
        """

        assert player == self.current_player, "Player took shot out of turn"
        assert params, "No params sent to PoolGame.take_shot()"
        assert all([kword in params.keys() for kword in LIMITS.keys()]), "Invalid params sent to PoolGame.take_shot()"
        
        foul = self.strike(
            V0=params["V0"],
            theta=params["theta"],
            phi=params["phi"],
            a=params["a"],
            b= params["b"],              
            check_rules=True, target_balls=self.target_balls[player]
        )

        if foul != Fouls.NONE:

            idx = self.players.index(player)
            self.current_player = self.players[(idx + 1) % 2]
            self.double_shot = True

            if foul == Fouls.POT_CUE_BALL:
                self.reset_cue_ball()

            return False

        else: 

            events = self.get_events()
            if not any([Event.ball_pocket(ball) in events for ball in self.target_balls[player]]):

                if not self.double_shot:
                    # Player did not pot a ball, therefore turn must end 
                    # print(f"Player {player} failed to pot a ball, changing turn...")
                    idx = self.players.index(player)
                    self.current_player = self.players[(idx + 1) % 2]
            
            self.double_shot = False

            return True
    
    def check_player_win(self, player) -> bool:
        """Checks if all of the target balls of the player are potted.

        Args:
            player (_type_): The player to check if they won

        Returns:
            bool: If the player won
        """

        assert player in self.players, "Checked win for a non-existant player"

        target_balls = self.target_balls[player]
        current_balls = self.shot.get_board_state()["balls"].keys()

        return not any([ball in current_balls for ball in target_balls])

    def check_win(self) -> bool:
        """Checks if either player won the game

        Returns:
            bool: If a player won
        """

        return any([self.check_player_win("one"), self.check_player_win("two")])

    def reward(self):
        """Called when game has ended, return a reward for either player

        Raises:
            Exception: If the game has not ended

        Returns:
            _type_: The reward for each player
        """

        if self.check_player_win("one"):
            return [1,0]
        
        if self.check_player_win("two"):
            return [0,1]
        
        raise Exception("PoolGame.reward() called during game")

    def get_value_estimate(self, get_shot_function, initial_roll_outs = 100) -> float:
        """Get the value estimate of the current state by performing a roll out

        Returns:
            float: The value estimate of the current state
        """

        max_depth = 6

        class Node:
            def __init__(self, state, player, double_shot, parent=None, action=None):
                self.state = state
                self.parent = parent
                self.children = []
                self.action = action
                self.player = player
                self.double_shot = double_shot

                self.value = 0
                self.visits = 0

            def expand(self, env, depth=1):   

                if depth > max_depth:
                    return

                branching_factor = initial_roll_outs / (10 ** (depth-1)) 
                branching_factor = 1 if branching_factor < 1 else int(branching_factor)

                env.from_state(self.state)
                if env.check_win():
                    reward = env.reward()[0]
                    self.backpropagate(reward)
                    return

                for _ in range(branching_factor):
                    env.current_player = self.player
                    env.double_shot = self.double_shot
                    action = get_shot_function(env)

                    try:
                        env.take_shot(self.player, action)
                    except:
                        # Can occasionally fail due to missing cue ball
                        continue

                    new_state = env.get_state()
                    new_state.params = action
                    child = Node(
                        new_state, 
                        env.current_player,
                        env.double_shot,
                        parent=self, 
                        action=action
                    )
                    self.children.append(child)

                for child in self.children:
                    child.expand(env, depth+1)

            def backpropagate(self, reward):
                node = self
                while True:
                    node.value += reward
                    node.visits += 1
                    if node.parent is None:
                        break
                    node = node.parent

        root = Node(
            self.get_state(),
            self.current_player,
            self.double_shot
        )

        root.expand(self)

        return root.value / root.visits

        
            

