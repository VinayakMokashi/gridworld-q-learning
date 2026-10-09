"""
The Grid World environment.

The world is a 5 x 8 grid (40 states). Every state is a zero-indexed
(row, col) tuple. Tuples are immutable, so they can be used directly as
dictionary keys - the agent's Q-table depends on this.

          col  0    1    2    3    4    5    6    7
    row 0      .    .    .   +3    .    .    .    .
    row 1      .    .    .    .    .  +20    .    .
    row 2      S    .    .    .    .    .    .    .
    row 3      .    .    .    .    .    .    .    .
    row 4      .    .  -10    .    .    .    .    .

S is the start ("your house"). The three numbered squares are terminal
states ("restaurants"): the moment you step onto one you collect its reward
and the episode ends. The next episode starts again from S.

The environment is the only thing that knows where you are and how much
reward to hand out. The agent only chooses actions.
"""

DEFAULT_GRID_SIZE = (5, 8)
DEFAULT_START = (2, 0)
DEFAULT_TERMINAL_REWARDS = {
    (0, 3): 3,     # an okay restaurant
    (1, 5): 20,    # the best restaurant
    (4, 2): -10,   # a terrible one - the punishment
}


class GridWorld:
    def __init__(self, grid_size=DEFAULT_GRID_SIZE, start_position=DEFAULT_START,
                 terminal_rewards=None, step_reward=0.0):
        self.grid_size = grid_size
        self.terminal_rewards = dict(terminal_rewards or DEFAULT_TERMINAL_REWARDS)

        # One step per move, no diagonals: four actions in every state, even at
        # the edges. Each effect is a (d_row, d_col) offset added to the state.
        self.actions = ["up", "down", "left", "right"]
        self.action_effects = {
            "up":    (-1, 0),
            "down":  (1, 0),
            "left":  (0, -1),
            "right": (0, 1),
        }

        # Reward for an ordinary (non-terminal) step. The default is 0; a value
        # of -1 turns "reach the goal" into "reach the goal by the shortest path".
        self.step_reward = step_reward

        if not self.in_bounds(start_position):
            raise ValueError(f"start position {start_position} is outside the {grid_size} grid")
        if start_position in self.terminal_rewards:
            raise ValueError(f"start position {start_position} is a terminal state")
        self.start_position = tuple(start_position)
        self.current_position = self.start_position

    @property
    def states(self):
        """Every (row, col) in the grid, terminal states included."""
        rows, cols = self.grid_size
        return [(r, c) for r in range(rows) for c in range(cols)]

    def in_bounds(self, state):
        rows, cols = self.grid_size
        return 0 <= state[0] < rows and 0 <= state[1] < cols

    def is_terminal(self, state):
        return state in self.terminal_rewards

    def reset(self):
        """Back to the house: start a new day (episode)."""
        self.current_position = self.start_position
        return self.current_position

    def next_state(self, state, action):
        """Where `action` takes you from `state`. Walking into a wall leaves you in place."""
        d_row, d_col = self.action_effects[action]
        rows, cols = self.grid_size
        row = min(max(state[0] + d_row, 0), rows - 1)
        col = min(max(state[1] + d_col, 0), cols - 1)
        return (row, col)

    def step(self, action):
        """Apply `action` and return (next_state, reward, done).

        `done` is True when you have reached one of the terminal states.
        """
        self.current_position = self.next_state(self.current_position, action)

        if self.current_position in self.terminal_rewards:
            return self.current_position, self.terminal_rewards[self.current_position], True
        return self.current_position, self.step_reward, False
