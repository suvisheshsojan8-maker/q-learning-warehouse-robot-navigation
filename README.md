# Q-Learning Based Autonomous Warehouse Robot Navigation

## Project Overview
This project implements a simulation-based autonomous warehouse robot navigation system using Q-learning for path planning and obstacle avoidance. The robot learns to navigate a 10×10 grid from a start position to a goal while avoiding fixed obstacles.

## Objectives
- Implement Q-learning for autonomous path planning.
- Learn a navigation policy through rewards and penalties.
- Avoid obstacles in a simulated warehouse environment.
- Replan a safe path when dynamic obstacles are introduced.

## Features
- 10×10 grid-based warehouse simulation.
- Q-learning with four possible actions: Up, Right, Down, and Left.
- Training over 5,000 episodes.
- Reward visualization using Matplotlib.
- Dynamic obstacle addition through the graphical interface.
- Breadth-First Search (BFS) for fallback path planning and dynamic replanning.

## Technologies Used
- Python
- NumPy
- Matplotlib
- Q-learning (Reinforcement Learning)
- Breadth-First Search (BFS)

## Requirements
Install the required Python libraries:

```bash
pip install numpy matplotlib
```

## How to Run
1. Install Python 3.
2. Install the required libraries.
3. Download or clone this repository.
4. Open a terminal in the project folder.
5. Run:

```bash
python simulation.py
```

## Project Limitations
This project is a software simulation and does not control a physical warehouse robot. Dynamic obstacle handling uses BFS replanning without retraining the Q-learning model.

## Future Scope
Future improvements may include physical robot integration, sensor-based obstacle detection, and testing in a real warehouse environment.

## Author
Suvishesh Sojan
