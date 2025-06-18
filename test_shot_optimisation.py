import json
import random

from rich import print
from rich.table import Table
from rich.panel import Panel
import numpy as np

from poolagent import Pool, State, Event, PoolSolver

# 
# Shot optimisation test using the Simulated Annealing Optimiser (with no neural network guidance)
#

# Pool simulation creation and random state
pool = Pool()
state = State(random=True)
pool.from_state(state)
solver = PoolSolver()

# Show the initial state
pool.get_image().show_image()

# Create a list of target events for the shot search
BALLS=['red', 'blue', 'yellow']
POCKETS=['rt', 'rb', 'lb', 'lt', 'rc', 'lc']

TARGET_EVENTS = []
TARGET_BALL = random.choice(BALLS)
TARGET_POCKET = random.choice(POCKETS)

TARGET_EVENTS.append(Event.ball_cushion('cue'))
TARGET_EVENTS.append(Event.ball_collision('cue', TARGET_BALL))
TARGET_EVENTS.append(Event.ball_cushion(TARGET_BALL))
TARGET_EVENTS.append(Event.ball_cushion(TARGET_BALL, TARGET_POCKET))

# Perform the shot optimisation
shot, new_state, new_events, rating, all_results, foul = solver.get_shot(pool, state, TARGET_EVENTS, BALLS)

# Print the results
table = Table(title="Shot Optimisation Results", show_lines=True)
table.add_column("Parameter", style="bold cyan")
table.add_column("Value", style="bold")

shot_str = "\n".join([
    f"V0: {shot['V0']:.2f}",
    f"phi: {shot['phi']:.2f}",
    f"theta: {shot['theta']:.2f}",
    f"a: {shot['a']:.2f}",
    f"b: {shot['b']:.2f}"
])
table.add_row("Shot", shot_str)
table.add_row("Target Events", "\n".join([e.encoding for e in TARGET_EVENTS]))
table.add_row("True Events", "\n".join([e.encoding for e in new_events]))
table.add_row("Rating", f"{rating:.3f}")
table.add_row("Foul", str(foul))

print(Panel.fit(table, title="Summary", border_style="magenta"))

# Save the shot gif
pool.save_shot_gif(state, shot, "test_shot.gif")