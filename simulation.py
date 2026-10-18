import random
from collections import deque

import numpy as np
import matplotlib

# Use Tkinter for the interactive Matplotlib window
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import matplotlib.animation as animation


# ============================================================
# 1. WAREHOUSE ENVIRONMENT
# ============================================================

GRID_SIZE = 10

START = (1, 1)
GOAL = (8, 8)

OBSTACLES = {
    (3, 3), (3, 4), (3, 5),
    (4, 5), (5, 5),
    (6, 2), (6, 3), (6, 4),
    (2, 7), (3, 7),
    (5, 2), (7, 5)
}

# Actions: Up, Right, Down, Left
ACTIONS = [
    (0, 1),
    (1, 0),
    (0, -1),
    (-1, 0)
]

ACTION_NAMES = ["Up", "Right", "Down", "Left"]


# ============================================================
# 2. Q-LEARNING PARAMETERS
# ============================================================

LEARNING_RATE = 0.1
DISCOUNT_FACTOR = 0.95

EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

EPISODES = 5000
MAX_STEPS = 200

GOAL_REWARD = 100
STEP_PENALTY = -1
OBSTACLE_PENALTY = -20


# ============================================================
# 3. ENVIRONMENT FUNCTIONS
# ============================================================

def is_valid_position(position):
    """Check whether a position is inside the grid."""
    x, y = position

    return (
        0 <= x < GRID_SIZE
        and 0 <= y < GRID_SIZE
    )


def is_valid_move(position):
    """Check whether a position is valid and not an obstacle."""
    return (
        is_valid_position(position)
        and position not in OBSTACLES
    )


def get_next_state(state, action):
    """Calculate the next position after an action."""
    dx, dy = ACTIONS[action]

    next_state = (
        state[0] + dx,
        state[1] + dy
    )

    if not is_valid_position(next_state):
        return state

    if next_state in OBSTACLES:
        return state

    return next_state


def get_reward(state):
    """Return the reward for the current state."""
    if state == GOAL:
        return GOAL_REWARD

    return STEP_PENALTY


# ============================================================
# 4. Q-LEARNING TRAINING
# ============================================================

def train_agent():
    global EPSILON

    q_table = np.zeros(
        (GRID_SIZE, GRID_SIZE, len(ACTIONS))
    )

    episode_rewards = []

    EPSILON = 1.0

    print("\nTraining Q-learning agent...")

    for episode in range(EPISODES):

        state = START
        total_reward = 0

        for step in range(MAX_STEPS):

            # Epsilon-greedy action selection
            if random.random() < EPSILON:
                action = random.randint(
                    0, len(ACTIONS) - 1
                )
            else:
                action = int(
                    np.argmax(
                        q_table[state[0], state[1]]
                    )
                )

            next_state = get_next_state(
                state, action
            )

            # Reward calculation
            if next_state == GOAL:
                reward = GOAL_REWARD
            elif next_state == state:
                reward = OBSTACLE_PENALTY
            else:
                reward = STEP_PENALTY

            # Q-learning update
            old_value = q_table[
                state[0], state[1], action
            ]

            next_max = np.max(
                q_table[
                    next_state[0],
                    next_state[1]
                ]
            )

            new_value = old_value + LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * next_max
                - old_value
            )

            q_table[
                state[0], state[1], action
            ] = new_value

            state = next_state
            total_reward += reward

            if state == GOAL:
                break

        episode_rewards.append(total_reward)

        # Reduce exploration over time
        EPSILON = max(
            EPSILON_MIN,
            EPSILON * EPSILON_DECAY
        )

        if (episode + 1) % 500 == 0:
            print(
                f"Episode {episode + 1}/{EPISODES} "
                f"| Reward: {total_reward} "
                f"| Epsilon: {EPSILON:.3f}"
            )

    print("Training completed!")

    return q_table, episode_rewards


# ============================================================
# 5. PLOT TRAINING REWARDS
# ============================================================

def plot_rewards(episode_rewards):

    plt.figure(figsize=(10, 5))

    plt.plot(
        episode_rewards,
        alpha=0.35,
        label="Episode Reward"
    )

    window_size = 100

    if len(episode_rewards) >= window_size:
        moving_average = np.convolve(
            episode_rewards,
            np.ones(window_size) / window_size,
            mode="valid"
        )

        plt.plot(
            range(
                window_size - 1,
                len(episode_rewards)
            ),
            moving_average,
            linewidth=2,
            label="100-Episode Moving Average"
        )

    plt.title("Q-Learning Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show(block=True)


# ============================================================
# 6. EXTRACT A PATH FROM THE Q-TABLE
# ============================================================

def get_learned_path(q_table):

    state = START
    path = [state]
    visited = {state}

    for _ in range(GRID_SIZE * GRID_SIZE):

        if state == GOAL:
            return path

        q_values = q_table[
            state[0], state[1]
        ]

        # Try actions in descending Q-value order
        action_order = np.argsort(q_values)[::-1]

        next_state = None

        for action in action_order:

            candidate = get_next_state(
                state, int(action)
            )

            if candidate == state:
                continue

            if candidate in visited:
                continue

            next_state = candidate
            break

        if next_state is None:
            break

        state = next_state
        path.append(state)
        visited.add(state)

    # Use BFS if the learned path is incomplete
    if path[-1] != GOAL:
        print(
            "Using safe-path fallback for the initial route."
        )

        path = find_safe_path(
            START, set(OBSTACLES)
        )

    return path


# ============================================================
# 7. BFS SAFE-PATH PLANNING
# ============================================================

def find_safe_path(start, obstacles):

    queue = deque([[start]])
    visited = {start}

    while queue:

        path = queue.popleft()
        current = path[-1]

        if current == GOAL:
            return path

        for dx, dy in ACTIONS:

            next_position = (
                current[0] + dx,
                current[1] + dy
            )

            if not is_valid_position(next_position):
                continue

            if next_position in obstacles:
                continue

            if next_position in visited:
                continue

            visited.add(next_position)

            queue.append(
                path + [next_position]
            )

    return []


# ============================================================
# 8. LIVE ROBOT SIMULATION
# ============================================================

def live_simulation(initial_path):

    if not initial_path:
        print("No valid path found.")
        return

    live_obstacles = set(OBSTACLES)

    current_path = list(initial_path)
    robot_index = 0

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(-0.5, GRID_SIZE - 0.5)

    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))

    ax.set_xticks(
        np.arange(-0.5, GRID_SIZE, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, GRID_SIZE, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="gray",
        linestyle="-",
        linewidth=1
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.set_aspect("equal")

    ax.set_title(
        "Q-Learning Warehouse Robot Navigation\n"
        "Click an empty cell to add a new obstacle"
    )

    # Draw fixed obstacles
    for x, y in live_obstacles:
        ax.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                color="black",
                zorder=3
            )
        )

    # Draw start and goal
    ax.scatter(
        START[0], START[1],
        color="green",
        s=250,
        label="Start",
        zorder=5
    )

    ax.scatter(
        GOAL[0], GOAL[1],
        color="red",
        s=250,
        label="Goal",
        zorder=5
    )

    # Draw the planned path
    path_line, = ax.plot(
        [p[0] for p in current_path],
        [p[1] for p in current_path],
        color="blue",
        linestyle="--",
        linewidth=2,
        label="Planned Path",
        zorder=4
    )

    # Robot marker
    robot, = ax.plot(
        [START[0]],
        [START[1]],
        marker="o",
        color="orange",
        markersize=18,
        linestyle="None",
        label="Robot",
        zorder=6
    )

    status_text = ax.text(
        0.5,
        1.02,
        "Robot is moving toward the goal",
        transform=ax.transAxes,
        ha="center",
        fontsize=10
    )

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1)
    )

    # --------------------------------------------------------
    # Animation update
    # --------------------------------------------------------

    def update(frame):
        nonlocal robot_index

        if robot_index >= len(current_path) - 1:

            robot_position = current_path[-1]

            if robot_position == GOAL:
                status_text.set_text(
                    "Goal reached!"
                )

            return robot, path_line, status_text

        robot_index += 1

        robot_position = current_path[robot_index]

        robot.set_data(
            [robot_position[0]],
            [robot_position[1]]
        )

        status_text.set_text(
            f"Robot position: {robot_position}"
        )

        return robot, path_line, status_text

    # --------------------------------------------------------
    # Add a new obstacle by clicking the grid
    # --------------------------------------------------------

    def add_obstacle(event):
        nonlocal current_path

        if event.inaxes != ax:
            return

        if event.xdata is None or event.ydata is None:
            return

        x = int(round(event.xdata))
        y = int(round(event.ydata))

        new_obstacle = (x, y)

        if not (
            0 <= x < GRID_SIZE
            and 0 <= y < GRID_SIZE
        ):
            return

        robot_position = current_path[robot_index]

        if new_obstacle in (
            START,
            GOAL,
            robot_position
        ):
            print(
                "Cannot place an obstacle on the "
                "robot, start, or goal."
            )
            return

        if new_obstacle in live_obstacles:
            print("That cell is already blocked.")
            return

        # Check whether a safe route still exists
        test_obstacles = (
            live_obstacles | {new_obstacle}
        )

        new_route = find_safe_path(
            robot_position,
            test_obstacles
        )

        if not new_route:
            print(
                "No safe route available. "
                "Obstacle not added."
            )
            return

        # Add the new obstacle
        live_obstacles.add(new_obstacle)

        ax.add_patch(
            plt.Rectangle(
                (x - 0.5, y - 0.5),
                1,
                1,
                color="black",
                zorder=4
            )
        )

        # Keep the travelled path and append the new route
        current_path = (
            current_path[:robot_index]
            + new_route
        )

        # Update the displayed path
        path_line.set_data(
            [p[0] for p in current_path],
            [p[1] for p in current_path]
        )

        status_text.set_text(
            f"Obstacle at {new_obstacle}! Rerouting..."
        )

        print(
            f"\nNew obstacle added at: {new_obstacle}"
        )

        print("New route:", current_path)

        fig.canvas.draw_idle()

    # Connect mouse clicks to the obstacle function
    fig.canvas.mpl_connect(
        "button_press_event",
        add_obstacle
    )

    # --------------------------------------------------------
    # Start animation
    # --------------------------------------------------------

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=1000,
        interval=1000,
        blit=False,
        repeat=False
    )

    plt.tight_layout()
    plt.show(block=True)


# ============================================================
# 9. MAIN PROGRAM
# ============================================================

def main():

    print("=" * 55)
    print("Q-LEARNING WAREHOUSE ROBOT NAVIGATION")
    print("=" * 55)

    q_table, episode_rewards = train_agent()

    # Display training performance
    plot_rewards(episode_rewards)

    # Generate the initial path
    initial_path = get_learned_path(q_table)

    print("\nInitial robot path:")
    print(initial_path)

    print(
        "\nClose the reward graph to start the simulation."
    )

    print(
        "During the simulation, click an empty grid cell "
        "to add a new obstacle."
    )

    print(
        "The robot will calculate a new safe route."
    )

    live_simulation(initial_path)


if __name__ == "__main__":
    main()