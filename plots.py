"""
Matplotlib figures of the training run and the learned Q-table.

    training_curves.png    reward, steps, outcomes and epsilon per episode
    value_policy.png       V(s) heatmap with the learned policy and the greedy route
    q_table.png            every Q(s, a): each cell split into four triangles
    value_snapshots.png    V(s) at several points in training - the values spreading out
    value_propagation.gif  the same thing as an animation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Polygon

import palette

CMAP = LinearSegmentedColormap.from_list("q_diverging", palette.DIVERGING_STOPS)
VALUE_MIN, VALUE_MAX = -10.0, 20.0
ARROW_OFFSETS = {"up": (0, -1), "down": (0, 1), "left": (-1, 0), "right": (1, 0)}  # (dx, dy) on screen

plt.rcParams.update({
    "figure.facecolor": palette.SURFACE,
    "axes.facecolor": palette.SURFACE,
    "savefig.facecolor": palette.SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10,
    "text.color": palette.INK,
    "axes.labelcolor": palette.INK_SECONDARY,
    "axes.edgecolor": palette.BASELINE,
    "axes.titlesize": 11,
    "axes.titleweight": "semibold",
    "axes.titlelocation": "left",
    "xtick.color": palette.INK_MUTED,
    "ytick.color": palette.INK_MUTED,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
})


def _norm():
    return TwoSlopeNorm(vmin=VALUE_MIN, vcenter=0.0, vmax=VALUE_MAX)


def _value_grid(env, q_table):
    """V(s) = max_a Q(s, a) as a 2-D array; terminal squares hold their reward."""
    grid = np.zeros(env.grid_size)
    for state in env.states:
        if env.is_terminal(state):
            grid[state] = env.terminal_rewards[state]
        else:
            grid[state] = max(q_table[state].values())
    return grid


def _draw_board(ax, env, values, show_numbers=False):
    """Heatmap of `values` with cell gaps, terminal labels and the start marker."""
    image = ax.imshow(values, cmap=CMAP, norm=_norm())
    rows, cols = env.grid_size
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color=palette.SURFACE, linewidth=2)
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for state, reward in env.terminal_rewards.items():
        bg = palette.diverging_color(reward, VALUE_MIN, VALUE_MAX)
        ax.text(state[1], state[0], f"{reward:+g}", ha="center", va="center",
                fontsize=12, fontweight="bold", color=palette.readable_ink(bg))

    sr, sc = env.start_position
    ax.text(sc - 0.42, sr - 0.42, "S", ha="left", va="top", fontsize=9,
            fontweight="bold", color=palette.START)

    if show_numbers:
        for state in env.states:
            if env.is_terminal(state):
                continue
            bg = palette.diverging_color(values[state], VALUE_MIN, VALUE_MAX)
            ax.text(state[1], state[0] + 0.3, f"{values[state]:.1f}", ha="center", va="center",
                    fontsize=7.5, color=palette.readable_ink(bg))
    return image


def _draw_policy_arrows(ax, env, q_table, values, size=14):
    """An arrow for the best action in every square that has learned something."""
    for state in env.states:
        if env.is_terminal(state):
            continue
        q_values = q_table[state]
        if all(q == 0 for q in q_values.values()):
            continue
        dx, dy = ARROW_OFFSETS[max(q_values, key=q_values.get)]
        bg = palette.diverging_color(values[state], VALUE_MIN, VALUE_MAX)
        r, c = state
        ax.annotate("", xy=(c + 0.28 * dx, r - 0.08 + 0.28 * dy), xytext=(c - 0.28 * dx, r - 0.08 - 0.28 * dy),
                    arrowprops=dict(arrowstyle="-|>", color=palette.readable_ink(bg), lw=1.6,
                                    mutation_scale=size), zorder=3)


def _moving_average(values, window):
    """Trailing average over the last `window` values (fewer at the very start)."""
    values = np.asarray(values, dtype=float)
    totals = np.cumsum(np.insert(values, 0, 0.0))
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(0, ends - window)
    return ends, (totals[ends] - totals[starts]) / (ends - starts)


def _episodes_label(episode):
    if episode == 0:
        return "before training"
    return f"after {episode} episode" + ("" if episode == 1 else "s")


def _finish(fig, path, show):
    fig.savefig(path, dpi=150, bbox_inches="tight")
    if not show:
        plt.close(fig)


# --------------------------------------------------------------------------- figures

def plot_training_curves(history, env, path, window=100, show=False):
    episodes = np.arange(1, len(history.episode_rewards) + 1)
    fig, axes = plt.subplots(4, 1, figsize=(10, 11), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 3, 1.6], "hspace": 0.45})
    blue = palette.OUTCOME_COLORS["+20"]

    ax = axes[0]
    ax.plot(*_moving_average(history.episode_rewards, window), color=blue, linewidth=2)
    ax.set_title(f"Average reward per episode ({window}-episode window)")
    ax.set_ylabel("reward")

    ax = axes[1]
    ax.plot(episodes, history.episode_steps, color=blue, alpha=0.18, linewidth=0.8, label="each episode")
    ax.plot(*_moving_average(history.episode_steps, window), color=blue, linewidth=2,
            label=f"{window}-episode average")
    ax.set_title("Steps taken per episode (fewer = a more direct route)")
    ax.set_ylabel("steps")
    ax.legend(loc="upper right")

    ax = axes[2]
    labels = {state: f"{reward:+g}" for state, reward in env.terminal_rewards.items()}
    order = sorted(env.terminal_rewards, key=lambda s: -env.terminal_rewards[s])
    names = [labels[s] for s in order] + ["timeout"]
    shares = []
    for state in order + [None]:
        hits = [1.0 if outcome == state else 0.0 for outcome in history.outcomes]
        x, share = _moving_average(hits, window)
        shares.append(share * 100)
    colors = [palette.OUTCOME_COLORS.get(name, palette.INK_MUTED) for name in names]
    pretty = [f"reached {n}" if n != "timeout" else "gave up (max steps)" for n in names]
    ax.stackplot(x, shares, colors=colors, labels=pretty, edgecolor=palette.SURFACE, linewidth=1)
    ax.set_ylim(0, 100)
    ax.set_title(f"Where the episodes ended ({window}-episode window)")
    ax.set_ylabel("% of episodes")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.08), ncol=4, fontsize=9)

    ax = axes[3]
    ax.plot(episodes, history.epsilons, color=blue, linewidth=2)
    ax.set_ylim(0, 1.05)
    ax.set_title("Exploration rate ε (chance of a random move)")
    ax.set_ylabel("ε")
    ax.set_xlabel("episode")

    for ax in axes[:2]:
        ax.grid(axis="y", color=palette.GRIDLINE, linewidth=0.8)
    for ax in axes:
        ax.set_xlim(1, len(episodes))
    _finish(fig, path, show)


def plot_value_policy(env, agent, greedy_path, path, show=False):
    values = _value_grid(env, agent.q_table)
    fig, ax = plt.subplots(figsize=(10, 6.6))
    image = _draw_board(ax, env, values, show_numbers=True)

    if len(greedy_path) > 1:
        xs = [c for _, c in greedy_path]
        ys = [r for r, _ in greedy_path]
        ax.plot(xs, ys, color=palette.INK, alpha=0.18, linewidth=14,
                solid_capstyle="round", solid_joinstyle="round", zorder=2)

    _draw_policy_arrows(ax, env, agent.q_table, values)

    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("V(s) = max over actions of Q(s, a)")
    colorbar.outline.set_visible(False)
    ax.set_title("Learned policy: arrow = best action, number = V(s), shaded band = greedy route from S",
                 pad=12)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    _finish(fig, path, show)


def plot_q_table(env, agent, path, show=False):
    """Each cell split into four triangles, one per action, coloured by Q(s, a)."""
    rows, cols = env.grid_size
    fig, ax = plt.subplots(figsize=(10, 6.6))
    norm = _norm()

    for state in env.states:
        r, c = state
        if env.is_terminal(state):
            reward = env.terminal_rewards[state]
            square = [(c - .5, r - .5), (c + .5, r - .5), (c + .5, r + .5), (c - .5, r + .5)]
            ax.add_patch(Polygon(square, facecolor=CMAP(norm(reward)), edgecolor=palette.SURFACE, linewidth=2))
            bg = palette.diverging_color(reward, VALUE_MIN, VALUE_MAX)
            ax.text(c, r, f"{reward:+g}", ha="center", va="center", fontsize=12,
                    fontweight="bold", color=palette.readable_ink(bg))
            continue

        corners = {
            "up":    [(c - .5, r - .5), (c + .5, r - .5), (c, r)],
            "down":  [(c - .5, r + .5), (c + .5, r + .5), (c, r)],
            "left":  [(c - .5, r - .5), (c - .5, r + .5), (c, r)],
            "right": [(c + .5, r - .5), (c + .5, r + .5), (c, r)],
        }
        q_values = agent.q_table[state]
        best = max(q_values, key=q_values.get) if any(q != 0 for q in q_values.values()) else None
        for action, triangle in corners.items():
            q = q_values[action]
            ax.add_patch(Polygon(triangle, facecolor=CMAP(norm(q)), edgecolor=palette.SURFACE, linewidth=1.5))
            if action == best:
                dx, dy = ARROW_OFFSETS[action]
                bg = palette.diverging_color(q, VALUE_MIN, VALUE_MAX)
                ax.plot(c + 0.3 * dx, r + 0.3 * dy, "o", markersize=4, color=palette.readable_ink(bg))

    sr, sc = env.start_position
    ax.text(sc - 0.46, sr - 0.46, "S", ha="left", va="top", fontsize=9, fontweight="bold", color=palette.START)
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    mappable = plt.cm.ScalarMappable(norm=norm, cmap=CMAP)
    colorbar = fig.colorbar(mappable, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Q(s, a)")
    colorbar.outline.set_visible(False)
    ax.set_title("The Q-table: each square is a state, each triangle an action (dot = best action)", pad=12)
    ax.set_xlabel("column")
    ax.set_ylabel("row")
    _finish(fig, path, show)


def _pick_snapshots(snapshots, count):
    """The first and last snapshot plus the ones nearest to log-spaced points in between."""
    episodes = sorted(snapshots)
    if len(episodes) <= count:
        return episodes
    last = episodes[-1]
    targets = np.geomspace(max(1, last / 200), last, count - 1)
    chosen = [episodes[0]]
    for target in targets:
        nearest = min(episodes, key=lambda e: abs(np.log(max(e, 1)) - np.log(target)))
        if nearest not in chosen:
            chosen.append(nearest)
    return chosen


def plot_value_snapshots(env, history, path, count=6, show=False):
    chosen = _pick_snapshots(history.snapshots, count)
    ncols = 3
    nrows = int(np.ceil(len(chosen) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.9 * nrows + 0.6), squeeze=False, layout="constrained")

    image = None
    for ax, episode in zip(axes.flat, chosen):
        image = _draw_board(ax, env, _value_grid(env, history.snapshots[episode]))
        ax.set_title(_episodes_label(episode), fontsize=10)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
    for ax in list(axes.flat)[len(chosen):]:
        ax.axis("off")

    colorbar = fig.colorbar(image, ax=axes, fraction=0.02, pad=0.02)
    colorbar.set_label("V(s)")
    colorbar.outline.set_visible(False)
    fig.suptitle("Values spread outward from the rewards as the agent gains experience",
                 x=0.05, ha="left", fontsize=12, fontweight="semibold")
    _finish(fig, path, show)


def make_value_gif(env, history, path, fps=4):
    episodes = sorted(history.snapshots)
    fig, ax = plt.subplots(figsize=(7.5, 5))

    def draw(frame):
        ax.clear()
        episode = episodes[frame]
        q_table = history.snapshots[episode]
        values = _value_grid(env, q_table)
        _draw_board(ax, env, values, show_numbers=True)
        _draw_policy_arrows(ax, env, q_table, values, size=10)
        ax.set_title(f"V(s) and best action, {_episodes_label(episode)}", pad=10)
        return []

    draw(0)

    frames = list(range(len(episodes))) + [len(episodes) - 1] * fps * 2   # hold the last frame
    anim = animation.FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(path, writer=animation.PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
