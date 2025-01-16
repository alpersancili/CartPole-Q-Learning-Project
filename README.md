# CartPole Q-Learning-Project
This project implements a Q-Learning agent to solve the classic CartPole environment provided by Gymnasium. The agent learns to balance a pole on a moving cart by discretizing the state space and using a Q-table to make optimal decisions.

PROJECT OVERVIEW

The CartPole problem is a reinforcement learning challenge where the goal is to keep a pole balanced upright on a cart by applying forces to move the cart left or right. This project uses Q-Learning, a model-free reinforcement learning algorithm, to train an agent to maximize its reward in this environment.
Key features of the project include:
Discretization of the state space into bins for efficient Q-table representation.
Epsilon-greedy strategy to balance exploration and exploitation.
Training and testing modes, with options to save and load the Q-table.
Visualization of training progress through a plot of rewards per episode.
Rendering of the environment during testing.

HOW IT WORKS

Environment Setup:
The CartPole environment is created using gymnasium. The continuous state space is divided into discrete bins for position, velocity, pole angle, and angular velocity.

Q-Learning Algorithm:

A 5D Q-table is initialized to store the expected rewards for each state-action pair.
During training, the agent uses the Q-learning update rule to iteratively improve the Q-values.
The epsilon-greedy strategy is applied to choose actions, gradually shifting from exploration to exploitation.

Training and Testing:

Training Mode: The agent interacts with the environment, updates the Q-table, and saves it to a file.
Testing Mode: The pre-trained Q-table is loaded, and the agent demonstrates its performance in the environment.

Reward Visualization:

The average rewards over the last 100 episodes are plotted to visualize the agent's learning progress.


Here's a professional description you can use to share your project on GitHub:

CartPole Q-Learning Project
This project implements a Q-Learning agent to solve the classic CartPole environment provided by Gymnasium. The agent learns to balance a pole on a moving cart by discretizing the state space and using a Q-table to make optimal decisions.

Project Overview
The CartPole problem is a reinforcement learning challenge where the goal is to keep a pole balanced upright on a cart by applying forces to move the cart left or right.

This project uses Q-Learning, a model-free reinforcement learning algorithm, to train an agent to maximize its reward in this environment.

Key features of the project include:

Discretization of the state space into bins for efficient Q-table representation.
Epsilon-greedy strategy to balance exploration and exploitation.
Training and testing modes, with options to save and load the Q-table.
Visualization of training progress through a plot of rewards per episode.
Rendering of the environment during testing.
How It Works
Environment Setup:
The CartPole environment is created using gymnasium. The continuous state space is divided into discrete bins for position, velocity, pole angle, and angular velocity.

Q-Learning Algorithm:

A 5D Q-table is initialized to store the expected rewards for each state-action pair.
During training, the agent uses the Q-learning update rule to iteratively improve the Q-values.
The epsilon-greedy strategy is applied to choose actions, gradually shifting from exploration to exploitation.
Training and Testing:

Training Mode: The agent interacts with the environment, updates the Q-table, and saves it to a file.
Testing Mode: The pre-trained Q-table is loaded, and the agent demonstrates its performance in the environment.
Reward Visualization:

The average rewards over the last 100 episodes are plotted to visualize the agent's learning progress.

PROJECT STRUCTURE

cartpole_qlearning.py: The main script containing the implementation of the Q-Learning algorithm.
cartpole.pkl: The saved Q-table generated after training (optional).
cartpole.png: A plot of average rewards over episodes (generated after training).

https://github.com/user-attachments/assets/129747f8-cc0c-4d4f-b568-a579f025e9f9

