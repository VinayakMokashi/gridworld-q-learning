"""
A tabular Q-learning agent.

The agent never sees the map. It only knows which actions exist, and it learns
from experience how much each action is worth in each state. That knowledge
lives in the Q-table.
"""

import copy
import random


class QLearningAgent:
    def __init__(self, states, actions, learning_rate=0.1, discount_factor=0.9,
                 epsilon=1.0, epsilon_decay=0.998, min_epsilon=0.1):
        self.actions = list(actions)
        self.learning_rate = learning_rate        # alpha: gradient-descent step size
        self.discount_factor = discount_factor    # gamma: how much the future is worth today

        # Probability of exploring. New in town, you try everything (epsilon = 1);
        # after each day you explore a little less, until you settle at
        # min_epsilon (10% by default). With epsilon fixed at 0.1 from day one,
        # the agent often finds the +3 restaurant first and keeps going back
        # to it without ever finding the +20 one. That is the explore-vs-exploit
        # dilemma. Pass epsilon=0.1, epsilon_decay=1.0 to see it.
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon

        # Q(s, a) as a dictionary of dictionaries: the outer key is the state s,
        # the inner key is the action a. Everything starts at zero because the
        # agent has never walked a path or visited a single restaurant.
        #
        #   >>> agent.q_table[(1, 4)]
        #   {'up': 0.0, 'down': 0.0, 'left': 0.0, 'right': 0.0}
        self.q_table = {state: {action: 0.0 for action in self.actions} for state in states}

    def get_best_action(self, state):
        """The action with the highest Q-value. Ties are broken at random."""
        q_values = self.q_table[state]
        best_value = max(q_values.values())
        best_actions = [action for action, q in q_values.items() if q == best_value]
        return random.choice(best_actions)

    def choose_action(self, state):
        """Epsilon-greedy: explore with probability epsilon, otherwise exploit."""
        if random.random() < self.epsilon:
            return random.choice(self.actions)    # EXPLORE: try a new restaurant
        return self.get_best_action(state)        # EXPLOIT: go back to the best one known

    def update_q_table(self, state, action, reward, next_state, done=False):
        """The Q-learning (temporal-difference) update:

            Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

        This is gradient descent on the squared Bellman error, treating the
        target r + gamma * max_a' Q(s', a') as a constant (the "semi-gradient").
        The bracketed term is the TD error. Nothing follows a terminal state, so
        there the target is just r.

        Returns the TD error, which shrinks towards zero as Q converges.
        """
        best_next_q = 0.0 if done else max(self.q_table[next_state].values())
        td_target = reward + self.discount_factor * best_next_q
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.learning_rate * td_error
        return td_error

    def decay_epsilon(self):
        """Explore a little less tomorrow than today. Called once per episode."""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def state_value(self, state):
        """V(s) = max_a Q(s, a): what the state is worth if you act greedily from it."""
        return max(self.q_table[state].values())

    def snapshot(self):
        """A frozen copy of the Q-table, used to watch the values spread during training."""
        return copy.deepcopy(self.q_table)
