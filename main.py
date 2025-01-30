import gymnasium as gym
import numpy as np
import cv2
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# Create the CartPole environment with rendering enabled
env = gym.make("CartPole-v1", render_mode="rgb_array")  # 'rgb_array' for manual rendering control

# Wrap the environment to log episode rewards
env = Monitor(env)

# Initialize PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# Training parameters
num_episodes = 1000  # Number of episodes to visualize before full training
total_timesteps = 50000

# Custom training loop with rendering and attempt count
print("Training the agent with visualization...")

for episode in range(num_episodes):
    obs, _ = env.reset()
    done = False
    episode_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)  # Predict action
        obs, reward, done, _, _ = env.step(action)  # Take action in the environment
        episode_reward += reward

        # Get the rendered frame from the environment
        frame = env.render()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV

        # Get frame dimensions
        height, width, _ = frame.shape
        
        # Add attempt text to the top-right corner
        text_position = (width - 250, 50)
        frame = cv2.putText(frame, f"Attempt: {episode + 1}", 
                            text_position, cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the updated frame in a window
        cv2.imshow("CartPole Training", frame)

        # Allow early exit by pressing 'q'
        if cv2.waitKey(50) & 0xFF == ord('q'):
            break

    print(f"Episode {episode + 1} finished with total reward: {episode_reward}")

    # Close the window after each attempt
    cv2.destroyAllWindows()

print("Continuing training without visualization...")
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("cartpole_ppo_trained")
print("Model saved successfully!")
