from environments import Projecitions
from datapreper import get_data
import numpy as np

trainX, trainy, _, _, _, _ = get_data()


env = Projecitions(X=trainX,y=trainy)
episodes = 20
rewards = []
for episode in range(episodes):
    iter_rewards = []

    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        obs, reward, done, info = env.step(random_action)
        iter_rewards.append(reward)

    rewards.append(iter_rewards)

print('Episodes Ran: ', episodes)
print('Episode Reward Mean: ',np.mean(rewards))
print('Max Potential Score: ', len(trainX)*100)
print('Minimum Potential Score: ',len(trainX)*(-5))