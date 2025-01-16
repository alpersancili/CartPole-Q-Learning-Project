# Importing necessary libraries
import gymnasium as gym  # For creating and interacting with the CartPole environment
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting the rewards graph
import pickle  # For saving and loading the Q-table

# Define the main function for running the Q-learning algorithm
def run(is_training=True, render=False):
    """
    Parameters:
    - is_training (bool): Determines if the model should train or use a pre-trained Q-table.
    - render (bool): If True, visualizes the environment during execution.
    """
    # Create the CartPole environment
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    # Discretize the continuous state space into bins for easier handling
    pos_space = np.linspace(-2.4, 2.4, 10)  # Cart position bins
    vel_space = np.linspace(-4, 4, 10)  # Cart velocity bins
    ang_space = np.linspace(-0.2095, 0.2095, 10)  # Pole angle bins
    ang_vel_space = np.linspace(-4, 4, 10)  # Pole angular velocity bins

    # Initialize or load the Q-table
    if is_training:
        # Create a 5D Q-table initialized to zero
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        # Load the pre-trained Q-table from a file
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    # Define hyperparameters
    learning_rate_a = 0.1  # Learning rate (alpha)
    discount_factor_g = 0.99  # Discount factor (gamma)
    epsilon = 1  # Initial exploration rate (100% random actions)
    epsilon_decay_rate = 0.00001  # Rate at which epsilon decreases
    rng = np.random.default_rng()  # Random number generator for reproducibility

    # List to store rewards for each episode
    rewards_per_episode = []

    i = 0  # Episode counter

    # for i in range(episodes):
    while True:
        # Reset the environment at the start of an episode
        state = env.reset()[0]  # Get the initial state
        # Discretize the initial state values into bins
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False  # Flag to track if the episode has ended
        rewards = 0  # Total rewards for the current episode

        # Run one episode until the pole falls or max steps are reached
        while not terminated and rewards < 10000:
            # Choose an action based on exploration vs exploitation
            if is_training and rng.random() < epsilon:
                # Choose random action (0=drive left, 1=stay neutral, 2=drive right)
                action = env.action_space.sample()  # Explore: Choose a random action
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])  # Exploit: Choose the best action

            # Take the chosen action and observe the new state and reward
            new_state, reward, terminated, _, _ = env.step(action)

            # Discretize the new state
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            # Update the Q-value using the Q-learning formula
            if is_training:
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            # Update the state variables for the next step
            state = new_state
            state_p = new_state_p
            state_v = new_state_v
            state_a = new_state_a
            state_av = new_state_av

            rewards += reward  # Accumulate the rewards

            # Print rewards during testing for every 100 steps
            if not is_training and rewards % 100 == 0:
                print(f'Episode: {i}  Rewards: {rewards}')

        # Store the total rewards of the episode
        rewards_per_episode.append(rewards)

        # Calculate the average reward over the last 100 episodes
        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode)-100):])

        # Print progress during training
        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:.2f}  Mean Rewards: {mean_rewards:.1f}')

        # Stop training if the agent achieves a high average reward
        if mean_rewards > 1000:
            break

        # Decay epsilon to reduce exploration over time
        epsilon = max(epsilon - epsilon_decay_rate, 0)

        i += 1  # Increment the episode counter

    # Close the environment
    env.close()

    # Save the trained Q-table to a file
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Plot the moving average of rewards over episodes
    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(i)]
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

# Main block to execute the function
if __name__ == '__main__':
    # Uncomment one of the following lines:
    # Train the agent and save the Q-table
    # run(is_training=True, render=False)

    # Load a pre-trained Q-table and visualize the agent
    run(is_training=False, render=True)
