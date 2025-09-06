import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy

"""
Lunarlander-v3
"""
env = gym.make("LunarLander-v3")
observation, info = env.reset()

print(f"Initial State S_0 {observation}")
print(f"Initial Info: {info}")

for _ in range(20):
    # take random action
    print("**********",_ , "**********")
    action = env.action_space.sample()
    print("Action taken: ", action)
    
    # do this random action and get the observation / state
    observation, reward, terminated, truncated, info = env.step(action)
    print(f"Next state S_t+1: {observation}")
    print(f"Reward R_t+1: {reward}")
    print(f"Terminated: {terminated}")
    print(f"Info: {info}") 

    # end game
    if terminated or truncated:
        print("Environment is reset")
        observation, info = env.reset()
    
env.close()


"""
    Observation Space S
    
    x-coor
    y-coor
    x-speed
    y-speed
    angle
    angle-speed
    left-flag
    right-flag
"""
env.reset()
print("_____OBSERVATION SPACE_____")
print("Observation Space Shape ", env.observation_space.shape) # (8, )
print("Observation Space type ", type(env.observation_space)) # <class 'gymnasium.spaces.box.Box'>
print("Sample Observation ", env.observation_space.sample())  # [-2.4296045  -0.05641006 -8.371265   -8.018182    4.31707    -9.034657  0.5735951   0.30920067]

"""
Action Space A

A0: none
A1: left engine
A2: main engine
A3: right engine
"""
print("\n_____ACTION SPACE____")
print("Action Space Shape", env.action_space.n)  # 4
print("Action Space Sample", env.action_space.sample())

"""
PPO

Proximal Policy Optimization
"""
# stacking multiple environments
env = make_vec_env('LunarLander-v3', n_envs=16)

model = PPO("MlpPolicy", env, n_steps=1024, batch_size=64, n_epochs=4, gamma=0.999, ent_coef=0.01, verbose=1)   
model_name = "ppo-LunarLander-v3"

# train
model.learn(total_timesteps=1000000, progress_bar=True) 

# save 
model.save(f"src/unit1/ckpt/{model_name}")

# load
# model = PPO.load(f"src/unit1/ckpt/{model_name}")

"""
Evaluate the agent
"""
eval_env = Monitor(gym.make("LunarLander-v3", render_mode='rgb_array'))
mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
