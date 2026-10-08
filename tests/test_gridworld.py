import random

import pytest

from agent import QLearningAgent
from gridworld import GridWorld
from train import greedy_rollout, train_agent


# --------------------------------------------------------------------------- environment

def test_grid_has_40_states_and_4_actions():
    env = GridWorld()
    assert len(env.states) == 40
    assert env.actions == ["up", "down", "left", "right"]


def test_step_moves_one_square():
    env = GridWorld()
    env.reset()                                  # (2, 0)
    assert env.step("up") == ((1, 0), 0.0, False)
    assert env.step("right") == ((1, 1), 0.0, False)


def test_walking_into_a_wall_stays_put():
    env = GridWorld()
    assert env.next_state((0, 0), "up") == (0, 0)
    assert env.next_state((0, 0), "left") == (0, 0)
    assert env.next_state((4, 7), "down") == (4, 7)
    assert env.next_state((4, 7), "right") == (4, 7)


@pytest.mark.parametrize("position, action, terminal, reward", [
    ((1, 4), "right", (1, 5), 20),
    ((0, 2), "right", (0, 3), 3),
    ((4, 3), "left", (4, 2), -10),
])
def test_terminal_squares_pay_out_and_end_the_episode(position, action, terminal, reward):
    env = GridWorld()
    env.current_position = position
    assert env.step(action) == (terminal, reward, True)


def test_step_reward_applies_to_ordinary_steps_only():
    env = GridWorld(step_reward=-1)
    env.reset()
    assert env.step("right") == ((2, 1), -1, False)
    env.current_position = (1, 4)
    assert env.step("right") == ((1, 5), 20, True)


def test_reset_returns_to_start():
    env = GridWorld()
    env.step("up")
    assert env.reset() == (2, 0)
    assert env.current_position == (2, 0)


@pytest.mark.parametrize("start", [(9, 9), (1, 5)])
def test_invalid_start_is_rejected(start):
    with pytest.raises(ValueError):
        GridWorld(start_position=start)


# --------------------------------------------------------------------------- agent

def make_agent(**kwargs):
    env = GridWorld()
    return env, QLearningAgent(env.states, env.actions, **kwargs)


def test_q_table_is_a_dict_of_dicts_of_zeros():
    _, agent = make_agent()
    assert agent.q_table[(1, 4)] == {"up": 0.0, "down": 0.0, "left": 0.0, "right": 0.0}
    assert len(agent.q_table) == 40


def test_update_follows_the_q_learning_formula():
    _, agent = make_agent(learning_rate=0.5, discount_factor=0.9)
    agent.q_table[(1, 3)] = {"up": 1.0, "down": 2.0, "left": 0.0, "right": 4.0}
    agent.q_table[(1, 2)]["right"] = 3.0

    td_error = agent.update_q_table((1, 2), "right", 0.0, (1, 3))

    # target = 0 + 0.9 * max(1, 2, 0, 4) = 3.6;  error = 3.6 - 3 = 0.6;  new Q = 3 + 0.5 * 0.6
    assert td_error == pytest.approx(0.6)
    assert agent.q_table[(1, 2)]["right"] == pytest.approx(3.3)


def test_update_into_a_terminal_ignores_the_future():
    _, agent = make_agent(learning_rate=0.1)
    agent.q_table[(1, 5)]["up"] = 100.0           # should never be looked at
    agent.update_q_table((1, 4), "right", 20, (1, 5), done=True)
    assert agent.q_table[(1, 4)]["right"] == pytest.approx(2.0)


def test_greedy_choice_when_epsilon_is_zero():
    _, agent = make_agent(epsilon=0.0, min_epsilon=0.0)
    agent.q_table[(2, 0)]["up"] = 5.0
    assert all(agent.choose_action((2, 0)) == "up" for _ in range(50))


def test_epsilon_decays_but_not_below_the_floor():
    _, agent = make_agent(epsilon=1.0, epsilon_decay=0.5, min_epsilon=0.1)
    for _ in range(10):
        agent.decay_epsilon()
    assert agent.epsilon == pytest.approx(0.1)


# --------------------------------------------------------------------------- training

def test_training_finds_the_shortest_route_to_the_best_restaurant():
    random.seed(42)
    env, agent = make_agent()
    history = train_agent(env, agent, num_episodes=2000, snapshot_episodes=[0, 100])

    path, actions, total_reward = greedy_rollout(env, agent)

    assert path[-1] == (1, 5)
    assert total_reward == 20
    assert len(actions) == 6                       # |2 - 1| + |0 - 5|
    assert agent.get_best_action((1, 4)) == "right"
    assert min(agent.q_table[(4, 3)], key=agent.q_table[(4, 3)].get) == "left"
    assert len(history.episode_rewards) == 2000
    assert sorted(history.snapshots) == [0, 100, 2000]
