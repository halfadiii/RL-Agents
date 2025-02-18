import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

#create and wrap the environment
def make_env():
    return gym.make("MountainCar-v0", render_mode='human')

#vectorized environment
env = DummyVecEnv([make_env])

#PPO agent
model = PPO("MlpPolicy", env, verbose=1)

# parameters
num_episodes =1000
total_timesteps = 50000

print("Training the agent")

for episode in range(num_episodes):
    obs =env.reset()
    done =False
    episode_reward = 0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs,rewards,dones, _ = env.step(action)

        done = dones[0]
        episode_reward += rewards[0]

    print(f"Episode {episode + 1} finished with total reward: {episode_reward}")

model.learn(total_timesteps=total_timesteps)
model.save("mountaincar_ppo_trained")
print("Model saved successfully!")

#total attempts after training is complete
print(f"Total attempts taken for the agent to learn: {num_episodes}")
