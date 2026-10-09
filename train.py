"""
The training loop and the greedy "deployment" walk.

Training is two nested loops. The outer loop is the days (episodes): every
morning you leave the house again. The inner loop is the steps you take that
day, until you reach a restaurant or give up after `max_steps_per_episode`.
"""

from dataclasses import dataclass, field


@dataclass
class TrainingHistory:
    episode_rewards: list = field(default_factory=list)   # total reward collected each episode
    episode_steps: list = field(default_factory=list)     # steps taken each episode
    outcomes: list = field(default_factory=list)          # terminal state reached, or None on timeout
    epsilons: list = field(default_factory=list)          # exploration rate used in each episode
    snapshots: dict = field(default_factory=dict)         # episode number -> copy of the Q-table


def train_agent(env, agent, num_episodes=2000, max_steps_per_episode=100,
                snapshot_episodes=(), on_episode_end=None):
    """Run Q-learning for `num_episodes` episodes and record how each one went.

    `snapshot_episodes` lists episode numbers at which to copy the Q-table
    before the episode starts. The finished table is always stored under
    `num_episodes`. `on_episode_end(episode, history)` is called after each episode.
    """
    history = TrainingHistory()
    snapshot_episodes = set(snapshot_episodes)

    for episode in range(num_episodes):
        if episode in snapshot_episodes:
            history.snapshots[episode] = agent.snapshot()

        state = env.reset()
        total_reward, steps, done = 0, 0, False

        for _ in range(max_steps_per_episode):
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            if done:          # reached a restaurant: done for the day
                break

        history.episode_rewards.append(total_reward)
        history.episode_steps.append(steps)
        history.outcomes.append(state if done else None)
        history.epsilons.append(agent.epsilon)
        agent.decay_epsilon()

        if on_episode_end is not None:
            on_episode_end(episode, history)

    history.snapshots[num_episodes] = agent.snapshot()
    return history


def greedy_rollout(env, agent, max_steps=100):
    """Follow the learned policy with no exploration (deployment).

    Returns (path, actions, total_reward) where `path` starts at the start
    state and `actions[i]` is the move taken from `path[i]`.
    """
    state = env.reset()
    path, actions, total_reward = [state], [], 0

    for _ in range(max_steps):
        action = agent.get_best_action(state)
        state, reward, done = env.step(action)
        path.append(state)
        actions.append(action)
        total_reward += reward
        if done:
            break

    return path, actions, total_reward
