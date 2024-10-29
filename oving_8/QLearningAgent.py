import gymnasium as gym
import numpy as np
import random

class QLearningAgent:
    def __init__(self, env, alpha=0.05, gamma=0.95, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.001):
        self.env = env # Environment in which the agent operates
        self.alpha = alpha # Learning rate: how much the Q-value is updated
        self.gamma = gamma # Discount factor: how much future rewards are valued
        self.epsilon = epsilon # Exploration rate: probability of exploring a new action
        self.epsilon_min = epsilon_min # Minimum exploration rate
        self.epsilon_decay = epsilon_decay # Rate of decay for exploration

        # Set up Q-table with bins for discretization
        self.action_space_size = self.env.action_space.n  # Number of possible actions
        self.state_space_size = self.create_state_space_bins() # Size of the state space after discretization
        self.q_table = np.zeros(self.state_space_size + (self.action_space_size,)) # Q-table with dimensions (state_space_size, action_space_size)

    # Helper function to create bins for discretization
    def create_state_space_bins(self, num_bins=20):
        """Discretize the continuous state space into bins."""
        bins = [
            np.linspace(-4.8, 4.8, num_bins),   # Cart Position
            np.linspace(-4, 4, num_bins),       # Cart Velocity
            np.linspace(-0.418, 0.418, num_bins),  # Pole Angle
            np.linspace(-4, 4, num_bins)        # Pole Angular Velocity
        ]
        return tuple(len(b) - 1 for b in bins) # Return the number of bins for each state dimension

    # Helper function to discretize the state
    def discretize_state(self, state):
        """Map continuous state to discrete bins."""
        cart_pos, cart_vel, pole_angle, pole_vel = state
        cart_pos_bin = np.digitize(cart_pos, np.linspace(-4.8, 4.8, self.state_space_size[0])) - 1
        cart_vel_bin = np.digitize(cart_vel, np.linspace(-4, 4, self.state_space_size[1])) - 1
        pole_angle_bin = np.digitize(pole_angle, np.linspace(-0.418, 0.418, self.state_space_size[2])) - 1
        pole_vel_bin = np.digitize(pole_vel, np.linspace(-4, 4, self.state_space_size[3])) - 1
        return (cart_pos_bin, cart_vel_bin, pole_angle_bin, pole_vel_bin) # Return the binned state

    # Helper function to choose an action
    def choose_action(self, state):
        """Epsilon-greedy action selection."""
        if random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.q_table[state]) # Exploit: choose the best known action
        else:
            return self.env.action_space.sample() # Explore: choose a random action

    # Helper function to update the Q-table
    def update_q_table(self, state, action, reward, new_state):
        """Update Q-value using the Q-learning formula."""
        best_future_q = np.max(self.q_table[new_state]) # Maximum Q-value for the next state
        # Update Q-value for the current state-action pair
        self.q_table[state][action] = (1 - self.alpha) * self.q_table[state][action] + self.alpha * (reward + self.gamma * best_future_q)

    # Train the agent using Q-learning
    def train(self, num_episodes=5000, max_steps_per_episode=200):
        rewards_all_episodes = [] # List to store total rewards for each episode

        for episode in range(num_episodes):
            state, _ = self.env.reset() # Reset environment and get initial state
            state = self.discretize_state(state) # Discretize the state
            rewards_current_episode = 0 # Reward counter for the current episode

            for step in range(max_steps_per_episode):
                action = self.choose_action(state) # Select an action
                new_state, reward, done, truncated, _ = self.env.step(action) # Execute action
                new_state = self.discretize_state(new_state) # Discretize new state

                self.update_q_table(state, action, reward, new_state) # Update Q-table
                state = new_state # Move to the new state
                rewards_current_episode += reward # Accumulate reward

                if done or truncated:
                    break # Exit loop if the episode is done

            # Decay epsilon to reduce exploration over time
            self.epsilon = max(self.epsilon_min, self.epsilon * np.exp(-self.epsilon_decay * episode))
            rewards_all_episodes.append(rewards_current_episode) # Record total rewards

        return rewards_all_episodes # Return rewards for all episodes

    # Test the agent without exploration
    def test(self, num_episodes=10, max_steps_per_episode=200):
        """Run the agent without exploration to evaluate its performance."""
        for episode in range(num_episodes):
            state, _ = self.env.reset() # Reset the environment
            state = self.discretize_state(state) # Discretize the initial state
            total_reward = 0 # Initialize total reward for the episode

            for step in range(max_steps_per_episode):
                action = np.argmax(self.q_table[state]) # Choose the best action
                new_state, reward, done, truncated, _ = self.env.step(action) # Execute the action
                new_state = self.discretize_state(new_state) # Discretize new state
                total_reward += reward # Accumulate reward
                self.env.render() # Render the environment

                if done or truncated: # Exit loop if the episode is done
                    break
                state = new_state # Move to the new state

            print(f"Episode {episode + 1}: Total Reward = {total_reward}") # Print total reward

        self.env.close() # Close the environment after testing


# Set up CartPole environment without rendering for training
env = gym.make("CartPole-v1")  # No render_mode specified for training
agent = QLearningAgent(env)

# Train the agent and get rewards
rewards = agent.train(num_episodes=3000)

# Print rewards per episode
for episode, reward in enumerate(rewards):
    print(f"Episode {episode + 1}: Total Reward = {reward}")


# Set up CartPole environment with rendering enabled for testing
env = gym.make("CartPole-v1", render_mode="human")
agent.env = env  # Update agent's environment to the one with rendering

# Test the trained agent with rendering enabled
agent.test(num_episodes=10)
