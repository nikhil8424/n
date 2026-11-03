
import numpy as np
import gymnasium as gym

# Building the environment
env = gym.make("FrozenLake-v1", is_slippery=False)

# Defining parameters
epsilon = 0.9
total_episodes = 1000
max_steps = 100
alpha = 0.85
gamma = 0.95

# Initializing the Q-matrix
Q = np.zeros((
    env.observation_space.n,
    env.action_space.n
))


# Function to choose the next action
def choose_action(state):

    if np.random.uniform(0, 1) < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q[state, :])

    return action


# Function to update Q-value
def update(state, action, reward, state2, action2, done):

    predict = Q[state, action]

    if done:
        target = reward
    else:
        target = reward + gamma * Q[state2, action2]

    Q[state, action] = (
        Q[state, action]
        + alpha * (target - predict)
    )


# Starting SARSA learning
total_rewards = 0

for episode in range(total_episodes):

    state1, info = env.reset()

    action1 = choose_action(state1)

    episode_reward = 0

    for t in range(max_steps):

        # Taking action
        state2, reward, terminated, truncated, info = env.step(action1)

        done = terminated or truncated

        # Choosing next action
        if not done:
            action2 = choose_action(state2)
        else:
            action2 = None

        # Updating Q-value
        update(
            state1,
            action1,
            reward,
            state2,
            action2 if action2 is not None else 0,
            done
        )

        episode_reward += reward

        if done:
            break

        state1 = state2
        action1 = action2

    total_rewards += episode_reward

    if (episode + 1) % 100 == 0:
        print(
            "Episode:", episode + 1,
            "Reward:", episode_reward
        )


# Evaluating performance
print("\nTraining Completed!")

print(
    "Performance:",
    total_rewards / total_episodes
)

# Displaying Q-matrix
print("\nQ-Matrix:")
print(np.round(Q, 3))

env.close()
