"""
Coloured drawings of the grid world, printed straight to the terminal.

Colours use 24-bit ANSI escape codes. They are switched off automatically
when output is not a terminal (or when NO_COLOR is set), and the maps fall
back to plain text with borders.
"""

import os
import sys
import time

import palette

ARROWS = {"up": "↑", "down": "↓", "left": "←", "right": "→"}
CELL_WIDTH = 7
VALUE_RANGE = (-10.0, 20.0)   # fixed colour scale: the worst and best terminal rewards

_color_enabled = False


# --------------------------------------------------------------------------- setup

def setup_terminal(color="auto"):
    """Switch stdout to UTF-8 and decide whether to use colour ("auto", "always", "never")."""
    global _color_enabled
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    if color == "never" or (color == "auto" and os.environ.get("NO_COLOR")):
        _color_enabled = False
    elif color == "always":
        _color_enabled = True
    else:
        _color_enabled = sys.stdout.isatty()

    if _color_enabled and os.name == "nt":
        _enable_windows_ansi()
    return _color_enabled


def _enable_windows_ansi():
    """Older Windows consoles need virtual-terminal processing switched on for ANSI codes."""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)                   # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
    except Exception:
        pass


def paint(text, fg=None, bg=None, bold=False, dim=False):
    """Wrap `text` in ANSI colour codes (hex colours), or return it unchanged without colour."""
    if not _color_enabled:
        return text
    codes = []
    if bold:
        codes.append("1")
    if dim:
        codes.append("2")
    if fg:
        codes.append("38;2;{};{};{}".format(*palette.hex_to_rgb(fg)))
    if bg:
        codes.append("48;2;{};{};{}".format(*palette.hex_to_rgb(bg)))
    if not codes:
        return text
    return f"\033[{';'.join(codes)}m{text}\033[0m"


# --------------------------------------------------------------------------- text helpers

def banner(title, subtitle=""):
    width = 64
    line = "═" * width
    print()
    print(paint(line, fg="#3987e5"))
    print(paint(title.center(width), bold=True))
    if subtitle:
        print(paint(subtitle.center(width), dim=True))
    print(paint(line, fg="#3987e5"))


def section(title):
    print()
    print(paint(f"▌ {title}", fg="#3987e5", bold=True))
    print(paint("─" * 64, dim=True))


def note(text):
    print(paint(f"  {text}", dim=True))


def fmt_reward(value):
    return f"{value:+g}"


# --------------------------------------------------------------------------- grids

def _render_grid(env, cell_fn, height=3):
    """Draw the grid. `cell_fn(state)` returns (lines, background_hex, bold)."""
    rows, cols = env.grid_size
    gutter = "     "
    header = gutter + "".join(paint(f"{c:^{CELL_WIDTH}}", dim=True) for c in range(cols))
    out = [header]

    if _color_enabled:
        for r in range(rows):
            cells = [cell_fn((r, c)) for c in range(cols)]
            for line_no in range(height):
                label = f"  {r}  " if line_no == height // 2 else gutter
                parts = []
                for lines, bg, bold in cells:
                    text = lines[line_no] if line_no < len(lines) else ""
                    parts.append(paint(f"{text:^{CELL_WIDTH}}", fg=palette.readable_ink(bg), bg=bg, bold=bold))
                out.append(paint(label, dim=True) + "".join(parts))
    else:
        rule = gutter + "+" + "+".join("-" * (CELL_WIDTH - 1) for _ in range(cols)) + "+"
        out.append(rule)
        for r in range(rows):
            cells = [cell_fn((r, c)) for c in range(cols)]
            for line_no in range(height):
                label = f"  {r}  " if line_no == height // 2 else gutter
                parts = [f"{(lines[line_no] if line_no < len(lines) else ''):^{CELL_WIDTH - 1}}"
                         for lines, _, _ in cells]
                out.append(label + "|" + "|".join(parts) + "|")
            out.append(rule)
    return "\n".join(out)


def _terminal_cell(env, state):
    reward = env.terminal_rewards[state]
    bg = palette.diverging_color(reward, *VALUE_RANGE)
    return ["", fmt_reward(reward), ""], bg, True


def render_world(env, path=None, actions=None, agent_at=None):
    """The map: start, terminals, and optionally a walked path drawn as arrows."""
    steps = {}
    if path and actions:
        for state, action in zip(path, actions):
            steps[state] = ARROWS[action]

    def cell(state):
        if state == agent_at:
            return ["", "●", ""], palette.START, True
        if env.is_terminal(state):
            return _terminal_cell(env, state)
        if state in steps:
            mark = steps[state] if state != env.start_position else "S" + steps[state]
            return ["", mark, ""], palette.GRIDLINE, True
        if state == env.start_position:
            return ["", "S", ""], palette.START, True
        return ["", "·", ""], palette.MIDPOINT, False

    return _render_grid(env, cell)


def render_policy(env, agent):
    """The learned policy: best action as an arrow, V(s) underneath, coloured by V(s)."""
    def cell(state):
        if env.is_terminal(state):
            return _terminal_cell(env, state)
        value = agent.state_value(state)
        q_values = agent.q_table[state]
        if all(q == 0 for q in q_values.values()):
            arrow = "?"                           # never learned anything here
        else:
            arrow = ARROWS[max(q_values, key=q_values.get)]
        if state == env.start_position:
            arrow = "S" + arrow
        return [arrow, f"{value:.1f}", ""], palette.diverging_color(value, *VALUE_RANGE), False

    return _render_grid(env, cell)


def value_scale_legend():
    """A one-line colour key for the value scale."""
    low, high = VALUE_RANGE
    ticks = [low, low / 2, 0, high / 4, high / 2, 3 * high / 4, high]
    swatches = "".join(paint("   ", bg=palette.diverging_color(v, low, high)) for v in ticks)
    if not _color_enabled:
        return f"     value scale: {low:+g} (bad) ... 0 (unknown) ... {high:+g} (good)"
    return f"     {fmt_reward(low)} {swatches} {fmt_reward(high)}   red = bad, gray = 0 / unknown, blue = good"


def world_legend(env):
    parts = []
    for state, reward in sorted(env.terminal_rewards.items(), key=lambda kv: -kv[1]):
        bg = palette.diverging_color(reward, *VALUE_RANGE)
        parts.append(paint(f" {fmt_reward(reward):>3} ", fg=palette.readable_ink(bg), bg=bg, bold=True) + f" at {state}")
    parts.append(paint("  S  ", fg="#ffffff", bg=palette.START, bold=True) + f" start {env.start_position}")
    return "     " + "    ".join(parts)


# --------------------------------------------------------------------------- Q-values

def render_q_bars(agent, state, width=24):
    """All four Q(s, a) for one state as horizontal bars."""
    q_values = agent.q_table[state]
    biggest = max((abs(q) for q in q_values.values()), default=0) or 1.0
    best = max(q_values, key=q_values.get)
    worst = min(q_values, key=q_values.get)
    lines = []
    for action, q in q_values.items():
        filled = round(abs(q) / biggest * width)
        color = palette.POSITIVE_ARM[1] if q >= 0 else palette.NEGATIVE_ARM[1]
        bar = paint("█" * filled, fg=color) + paint("░" * (width - filled), dim=True)
        tag = ""
        if q != 0 and action == best:
            tag = paint("  ◀ best", fg=palette.POSITIVE_ARM[1], bold=True)
        elif q < 0 and action == worst:
            tag = paint("  ◀ worst", fg=palette.NEGATIVE_ARM[1], bold=True)
        lines.append(f"    {ARROWS[action]} {action:<6} {bar} {q:+7.2f}{tag}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- training progress

def print_progress(episode, num_episodes, history, best_terminal, window=100):
    """One line summarising the last `window` episodes."""
    done = episode + 1
    recent = slice(max(0, done - window), done)
    rewards = history.episode_rewards[recent]
    steps = history.episode_steps[recent]
    outcomes = history.outcomes[recent]

    found_best = sum(1 for o in outcomes if o == best_terminal) / len(outcomes)
    bar_width = 20
    filled = round(done / num_episodes * bar_width)
    bar = paint("█" * filled, fg="#3987e5") + paint("░" * (bar_width - filled), dim=True)
    avg_reward = sum(rewards) / len(rewards)
    avg_steps = sum(steps) / len(steps)

    print(f"  episode {done:>5}/{num_episodes} {bar}  "
          f"ε {history.epsilons[episode]:.2f}  │  "
          f"avg reward {avg_reward:+6.2f}  │  "
          f"avg steps {avg_steps:5.1f}  │  "
          f"best restaurant {found_best:4.0%}")


# --------------------------------------------------------------------------- animation

def animate_walk(env, path, actions, delay=0.4):
    """Replay a walk step by step, redrawing the map in place."""
    if not _color_enabled:
        print(render_world(env, path, actions))
        return

    frame_height = None
    for i, state in enumerate(path):
        frame = render_world(env, path[:i], actions[:i], agent_at=state)
        caption = f"     step {i}/{len(path) - 1}   at {state}"
        if i < len(actions):
            caption += f"   next: {actions[i]} {ARROWS[actions[i]]}"
        frame += "\n" + paint(caption.ljust(60), bold=True)
        if frame_height is not None:
            sys.stdout.write(f"\033[{frame_height}A")   # move the cursor back up and redraw
        print(frame)
        sys.stdout.flush()
        frame_height = frame.count("\n") + 1
        time.sleep(delay)
