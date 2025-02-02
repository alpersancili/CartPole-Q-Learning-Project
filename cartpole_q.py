import gymnasium as gym  # For creating and interacting with the CartPole environment
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For plotting the rewards graph
import pickle  # For saving and loading the Q-table

def run(is_training=True, render=False, episodes=5000):
    """
    Parameters:
    - is_training (bool): Determines if the model should train or use a pre-trained Q-table.
    - render (bool): If True, visualizes the environment during execution.
    - episodes (int): Number of episodes for training/testing.
    """
    env = gym.make('CartPole-v1', render_mode='human' if render else None)

    pos_space = np.linspace(-2.4, 2.4, 10)  # Cart position bins
    vel_space = np.linspace(-4, 4, 10)  # Cart velocity bins
    ang_space = np.linspace(-0.2095, 0.2095, 10)  # Pole angle bins
    ang_vel_space = np.linspace(-4, 4, 10)  # Pole angular velocity bins

    if is_training:
        q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, env.action_space.n))
    else:
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1  # Learning rate
    discount_factor_g = 0.99  # Discount factor
    epsilon = 1  # Initial exploration rate
    epsilon_decay_rate = 0.001  # Rate at which epsilon decreases
    rng = np.random.default_rng()

    rewards_per_episode = []

    for i in range(episodes):
        state = env.reset()[0]
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        rewards = 0

        while not terminated and rewards < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)

            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            rewards += reward

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[max(0, len(rewards_per_episode)-100):])

        if is_training and i % 100 == 0:
            print(f'Episode: {i} {rewards}  Epsilon: {epsilon:.2f}  Mean Rewards: {mean_rewards:.1f}')

        if mean_rewards > 1000:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)

    env.close()

    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    mean_rewards = [np.mean(rewards_per_episode[max(0, t-100):(t+1)]) for t in range(len(rewards_per_episode))]
    plt.plot(mean_rewards)
    plt.savefig(f'cartpole.png')

if __name__ == '__main__':
    # Train the agent and save the Q-table
    # run(is_training=True, render=False, episodes=5000)

    # Load a pre-trained Q-table and visualize the agent
    run(is_training=False, render=True, episodes=10)
