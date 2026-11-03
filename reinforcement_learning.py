
import numpy as np
import gymnasium as gym

env = gym.make("FrozenLake-v1", is_slippery=False)

n_observations = env.observation_space.n
n_actions = env.action_space.n

print("Observations:", n_observations)
print("Actions:", n_actions)

Q_table = np.zeros((n_observations, n_actions))

print("\nInitial Q-Table:")
print(Q_table)

n_episodes = 10000
max_iter_episode = 100

exploration_proba = 1
exploration_decreasing_decay = 0.001
min_exploration_proba = 0.01

g = 0.99
learning_rate = 0.1

rewards_per_episode = []

for e in range(n_episodes):

    current_state, info = env.reset()

    done = False
    total_episode_reward = 0

    for i in range(max_iter_episode):

        # Choose action using epsilon-greedy policy
        if np.random.uniform(0, 1) < exploration_proba:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q_table[current_state, :])

        # Perform action
        next_state, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        # Calculate maximum future Q-value
        if done:
            max_future_q = 0
        else:
            max_future_q = np.max(Q_table[next_state, :])

        # Update Q-table
        Q_table[current_state, action] = (
            (1 - learning_rate) * Q_table[current_state, action]
            + learning_rate * (
                reward + g * max_future_q
            )
        )

        total_episode_reward += reward

        if done:
            break

        current_state = next_state

    # Reduce exploration probability
    exploration_proba = max(
        min_exploration_proba,
        np.exp(-exploration_decreasing_decay * e)
    )

    rewards_per_episode.append(total_episode_reward)

    if (e + 1) % 1000 == 0:
        print(
            "Episode:", e + 1,
            "Reward:", total_episode_reward,
            "Epsilon:", round(exploration_proba, 3)
        )


# Display mean reward per 1000 episodes
print("\nMean reward per thousand episodes")

for i in range(10):

    mean_reward = np.mean(
        rewards_per_episode[1000 * i:1000 * (i + 1)]
    )

    print(
        f"{1000 * i} - {(i + 1) * 1000}:",
        "Mean episode reward:", round(mean_reward, 4)
    )


# Overall performance
print(
    "\nMean episode reward of all 10000 episodes:",
    round(np.mean(rewards_per_episode), 4)
)

# Display final Q-table
print("\nFinal Q-Table:")
print(np.round(Q_table, 3))


# Test trained agent
test_env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)

state, info = test_env.reset()
total_reward = 0

for step in range(max_iter_episode):

    action = np.argmax(Q_table[state, :])

    state, reward, terminated, truncated, info = test_env.step(action)

    total_reward += reward

    if terminated or truncated:
        break

print("\nTesting Result:")
print("Test Reward:", total_reward)

test_env.close()
env.close()
