import numpy as np
import gymnasium as gym
import tensorflow as tf
import random

# Create Environment
env = gym.make("FrozenLake-v1", is_slippery=False)

n_observations = int(env.observation_space.n)
n_actions = int(env.action_space.n)

print(f"Observations: {n_observations}")
print(f"Actions: {n_actions}")

# Neural Network Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(
        n_actions,
        activation='linear',
        input_shape=(n_observations,)
    )
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
    loss='mse'
)

# Convert state into one-hot tensor
def state_to_tensor(state, n_states):
    return tf.one_hot([state], n_states)

# Hyperparameters
n_episodes = 1000
max_iter_episode = 100
exploration_proba = 1.0
exploration_decreasing_decay = 0.005
min_exploration_proba = 0.01
g = 0.99

rewards_per_episode = []

# Training Phase
print("\nTraining Deep Q-Network with TensorFlow...")

for e in range(n_episodes):

    current_state, _ = env.reset()
    total_episode_reward = 0

    for i in range(max_iter_episode):

        state_tensor = state_to_tensor(
            current_state, n_observations
        )

        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < exploration_proba:
            action = int(env.action_space.sample())
        else:
            q_values = model(
                state_tensor, training=False
            ).numpy()
            action = int(np.argmax(q_values[0]))

        next_state, reward, done, truncated, _ = env.step(action)

        # Calculate target Q-values
        q_values = model(
            state_tensor, training=False
        ).numpy()

        if done or truncated:
            target = reward
        else:
            next_state_tensor = state_to_tensor(
                next_state, n_observations
            )
            max_next_q = np.max(
                model(next_state_tensor, training=False).numpy()[0]
            )
            target = reward + g * max_next_q

        # Update Q-value and train model
        q_values[0][action] = target
        model.train_on_batch(state_tensor, q_values)

        total_episode_reward += reward
        current_state = next_state

        if done or truncated:
            break

    # Decay exploration rate
    exploration_proba = max(
        min_exploration_proba,
        np.exp(-exploration_decreasing_decay * e)
    )

    rewards_per_episode.append(total_episode_reward)

# Print Performance Metrics
print("\nMean reward per 100 episodes:")

for i in range(10):
    start = 100 * i
    end = 100 * (i + 1)

    print(
        f"Episodes {start} - {end}: "
        f"{np.mean(rewards_per_episode[start:end]):.4f}"
    )

print(
    f"\nOverall Mean Reward: "
    f"{np.mean(rewards_per_episode):.4f}"
)

env.close()

# Watch the Trained Agent Play
print("\nShowing Trained Agent in GUI...")

eval_env = gym.make(
    "FrozenLake-v1",
    is_slippery=False,
    render_mode="human"
)

eval_env.metadata["render_fps"] = 4

state, _ = eval_env.reset()

for _ in range(max_iter_episode):

    eval_env.render()

    state_tensor = state_to_tensor(
        state, n_observations
    )

    q_values = model(
        state_tensor, training=False
    ).numpy()

    action = int(np.argmax(q_values[0]))

    state, reward, done, truncated, _ = eval_env.step(action)

    if done or truncated:
        eval_env.render()
        break

if reward == 1.0:
    print("Goal Reached!")

eval_env.close()