from Environment.environments import Projecitions
from datapreper import get_data

trainX, trainy, _, _, _, _ = get_data()

env = Projecitions(X=trainX,y=trainy)
episodes = 1
rewards = []
for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        random_action = env.action_space.sample()
        print('Action: ', random_action)
        obs, reward, done, info = env.step(random_action)
        print('Reward: ', reward)