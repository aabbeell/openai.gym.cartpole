import gym
import numpy as np
import matplotlib.pyplot as plt


lr_1to01 = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
lr_09to04 = [0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5]
lr = []
avg_list = []

env = gym.make('CartPole-v0')


def run_episode(env, parameters):

    observation = env.reset()
    totalreward = 0
    # for 200 timesteps
    for _ in range(200):

        #env.render()
        action = 0 if np.matmul(parameters, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        totalreward += reward
        if done:
            break
    return totalreward


def hillclimb(noise_scaling):

    episodes_per_update = 5
    parameters = np.random.rand(4) * 2 -1
    bestreward = 0

    # 2000 episodes
    for x in range(2000):
        newparams = parameters + (np.random.rand(4) * 2 - 1) * noise_scaling
        reward = run_episode(env, newparams)
        # print("%f - %d : reward %d best %d" % (noise_scaling, x, reward, bestreward))
        if reward > bestreward:
            bestreward = reward
            parameters = newparams
            if reward == 200:
                break
    return x

for x in range(60,110,5):
    list = []
    lr.append(x/100.)
    for _ in range(40):
        list.append(hillclimb(x/100.))
    avg_list.append(np.mean(list))
    print(list)

plt.plot(lr, avg_list)
plt.show()

