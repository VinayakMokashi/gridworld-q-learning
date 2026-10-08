"""
Q-learning in Grid World - run this file.

    python main.py                 # train, print the results, save the figures
    python main.py --animate       # also replay the learned route step by step
    python main.py --help          # every option

The output follows the order of the lecture: the world, the empty Q-table,
training, the learned policy, a check of the lecture's intuition, and the
final greedy walk ("deployment").
"""

import argparse
import csv
import os
import random
import time

import numpy as np

import terminal_view as tv
from agent import QLearningAgent
from gridworld import GridWorld
from train import greedy_rollout, train_agent


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Q-learning agent to find the best restaurant in a 5 x 8 grid world.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    learning = parser.add_argument_group("learning")
    learning.add_argument("--episodes", type=int, default=2000, help="number of training episodes (days)")
    learning.add_argument("--max-steps", type=int, default=100, help="give up on an episode after this many steps")
    learning.add_argument("--alpha", type=float, default=0.1, help="learning rate")
    learning.add_argument("--gamma", type=float, default=0.9, help="discount factor")
    learning.add_argument("--epsilon", type=float, default=1.0, help="starting exploration rate")
    learning.add_argument("--epsilon-decay", type=float, default=0.998,
                          help="multiply epsilon by this after every episode (1.0 = never decay)")
    learning.add_argument("--min-epsilon", type=float, default=0.1, help="epsilon never decays below this")

    world = parser.add_argument_group("world")
    world.add_argument("--start", type=int, nargs=2, default=[2, 0], metavar=("ROW", "COL"),
                       help="start square")
    world.add_argument("--step-reward", type=float, default=0.0,
                       help="reward for every ordinary step; try -1 to reward short routes")

    output = parser.add_argument_group("output")
    output.add_argument("--seed", type=int, default=42, help="random seed, for repeatable runs")
    output.add_argument("--out", default="outputs", help="folder for the figures and the Q-table CSV")
    output.add_argument("--no-plots", action="store_true", help="skip the matplotlib figures")
    output.add_argument("--show", action="store_true", help="also open the figures in windows")
    output.add_argument("--animate", action="store_true", help="replay the learned route in the terminal")
    output.add_argument("--color", choices=["auto", "always", "never"], default="auto",
                        help="coloured terminal output")
    return parser.parse_args()


def approach(env, terminal, action):
    """The square from which `action` steps onto `terminal`, if there is one."""
    d_row, d_col = env.action_effects[action]
    state = (terminal[0] - d_row, terminal[1] - d_col)
    if env.in_bounds(state) and not env.is_terminal(state):
        return state
    return None


def check(passed, text):
    mark = tv.paint("  ✔ ", fg="#0ca30c", bold=True) if passed else tv.paint("  ✘ ", fg="#d03b3b", bold=True)
    print(mark + text)


def describe_route(path, actions):
    parts = [str(path[0])]
    for action, state in zip(actions, path[1:]):
        parts.append(f"{tv.ARROWS[action]} {state}")
    return "  ".join(parts)


def save_q_table_csv(env, agent, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["row", "col", "terminal_reward", *env.actions, "best_action", "value"])
        for state in env.states:
            if env.is_terminal(state):
                writer.writerow([*state, env.terminal_rewards[state], *[""] * len(env.actions), "", ""])
                continue
            q_values = agent.q_table[state]
            writer.writerow([*state, "", *[f"{q_values[a]:.4f}" for a in env.actions],
                             max(q_values, key=q_values.get), f"{agent.state_value(state):.4f}"])


def save_figures(env, agent, history, path, out_dir, show):
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import plots

    figures = [
        ("training_curves.png", lambda p: plots.plot_training_curves(history, env, p, show=show)),
        ("value_policy.png", lambda p: plots.plot_value_policy(env, agent, path, p, show=show)),
        ("q_table.png", lambda p: plots.plot_q_table(env, agent, p, show=show)),
        ("value_snapshots.png", lambda p: plots.plot_value_snapshots(env, history, p, show=show)),
        ("value_propagation.gif", lambda p: plots.make_value_gif(env, history, p)),
    ]
    for name, draw in figures:
        target = os.path.join(out_dir, name)
        draw(target)
        print(f"    saved {tv.paint(target, bold=True)}")
    if show:
        plt.show()


def main():
    args = parse_args()
    tv.setup_terminal(args.color)
    random.seed(args.seed)

    env = GridWorld(start_position=tuple(args.start), step_reward=args.step_reward)
    agent = QLearningAgent(env.states, env.actions, learning_rate=args.alpha, discount_factor=args.gamma,
                           epsilon=args.epsilon, epsilon_decay=args.epsilon_decay, min_epsilon=args.min_epsilon)
    best_terminal = max(env.terminal_rewards, key=env.terminal_rewards.get)
    worst_terminal = min(env.terminal_rewards, key=env.terminal_rewards.get)

    tv.banner("Q-LEARNING IN GRID WORLD", "Hands-on Introduction to Reinforcement Learning")

    # 1. The world --------------------------------------------------------------
    tv.section("1. The world")
    print(tv.render_world(env))
    print()
    print(tv.world_legend(env))
    print()
    rows, cols = env.grid_size
    tv.note(f"{rows} x {cols} grid = {rows * cols} states. 4 actions: up, down, left, right.")
    tv.note("Walking into a wall leaves you where you are. A numbered square ends the episode.")
    tv.note(f"Reward for an ordinary step: {args.step_reward:g}")
    print()
    print(f"    learning rate α = {args.alpha}    discount γ = {args.gamma}    "
          f"ε = {args.epsilon} → {args.min_epsilon} (×{args.epsilon_decay} per episode)")
    print(f"    episodes = {args.episodes}    max steps per episode = {args.max_steps}    seed = {args.seed}")

    # 2. Before training ----------------------------------------------------------
    tv.section("2. Before training: the Q-table is all zeros")
    lecture_state = approach(env, best_terminal, "right") or env.start_position
    tv.note("Q(s, a) is a dictionary of dictionaries: q_table[state][action].")
    tv.note("The agent has never walked a path, so it has no idea yet - even next to the +20.")
    print()
    print(f"    agent.q_table[{lecture_state}] = {agent.q_table[lecture_state]}")

    # 3. Training -----------------------------------------------------------------
    tv.section("3. Training (explore, then exploit)")
    tv.note("Each episode: start at S, choose actions epsilon-greedily, update Q after every step,")
    tv.note(f"stop at a numbered square or after {args.max_steps} steps. Averages are over the last 100 episodes.")
    print()

    checkpoint = max(1, args.episodes // 10)
    snapshot_episodes = sorted({0, *np.unique(np.geomspace(1, args.episodes, 40).astype(int)).tolist()})
    snapshot_episodes = [e for e in snapshot_episodes if e < args.episodes]

    def on_episode_end(episode, history):
        if (episode + 1) % checkpoint == 0 or episode + 1 == args.episodes:
            tv.print_progress(episode, args.episodes, history, best_terminal)

    started = time.perf_counter()
    history = train_agent(env, agent, args.episodes, args.max_steps, snapshot_episodes, on_episode_end)
    print()
    tv.note(f"trained {args.episodes} episodes in {time.perf_counter() - started:.2f} s")

    # 4. What was learned -----------------------------------------------------------
    tv.section("4. The learned policy")
    print(tv.render_policy(env, agent))
    print()
    print(tv.value_scale_legend())
    print()
    tv.note("Arrow = best action in that square.  Number = V(s) = max_a Q(s, a), the discounted")
    tv.note("reward the agent expects from there.  ? = a square it never learned anything about.")

    # 5. The lecture's intuition --------------------------------------------------
    tv.section("5. Checking the lecture's intuition")
    near_best = approach(env, best_terminal, "right")
    if near_best:
        print(f"  At {near_best}, right next to the {env.terminal_rewards[best_terminal]:+g}, "
              f"the best action should be RIGHT:")
        print(tv.render_q_bars(agent, near_best))
        check(agent.get_best_action(near_best) == "right", f"best action at {near_best} is right")
        print()

    near_worst = approach(env, worst_terminal, "left")
    if near_worst:
        q = agent.q_table[near_worst]
        print(f"  At {near_worst}, LEFT steps onto the {env.terminal_rewards[worst_terminal]:+g}, "
              f"so it should be the worst action:")
        print(tv.render_q_bars(agent, near_worst))
        check(min(q, key=q.get) == "left" and q["left"] < 0, f"worst action at {near_worst} is left")
        print()

    tv.note("Values are high near the +20, low near the -10, and fade with distance:")
    tv.note("every step away costs a factor of γ.")

    # 6. Deployment ---------------------------------------------------------------
    tv.section("6. Deployment: follow the policy with no exploration")
    path, actions, total_reward = greedy_rollout(env, agent, args.max_steps)
    if args.animate:
        tv.animate_walk(env, path, actions)
    else:
        print(tv.render_world(env, path, actions))
    print()
    print("    " + describe_route(path, actions))
    print()

    shortest = abs(best_terminal[0] - env.start_position[0]) + abs(best_terminal[1] - env.start_position[1])
    final = path[-1]
    if final == best_terminal:
        check(True, f"reached the best restaurant ({env.terminal_rewards[final]:+g}) in {len(actions)} steps "
                    f"(shortest possible: {shortest}), total reward {total_reward:+g}")
    elif env.is_terminal(final):
        check(False, f"settled for {env.terminal_rewards[final]:+g} in {len(actions)} steps - "
                     f"try more episodes or more exploration")
    else:
        check(False, f"did not reach a restaurant within {args.max_steps} steps")

    # 7. Outputs ------------------------------------------------------------------
    tv.section("7. Saved outputs")
    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "q_table.csv")
    save_q_table_csv(env, agent, csv_path)
    print(f"    saved {tv.paint(csv_path, bold=True)}")
    if args.no_plots:
        tv.note("figures skipped (--no-plots)")
    else:
        save_figures(env, agent, history, path, args.out, args.show)
    print()


if __name__ == "__main__":
    main()
