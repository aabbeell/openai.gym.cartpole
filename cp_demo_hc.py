import gym
import numpy as np

env = gym.make('CartPole-v0')

def run_episode(env, parameters):

    observation = env.reset()
    totalreward = 0
    timesteps = 3000
    for _ in range(timesteps):

        #env.render()

        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward

        if totalreward == 3000:
            break

    return totalreward

def hillclimb(noise_scaling=0.5):

    parameters = np.random.rand(4) * 2 -1
    bestreward = 0

    episodes = 2000

    for episode in range(episodes):
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_episode(env, newparams)
        print("noise_scaling: %f - episode : %d - reward : %d - best : %d" % (noise_scaling, episode, reward, bestreward))
        
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 3000:
                break
    return episode

hillclimb()