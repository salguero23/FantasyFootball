import numpy as np
from gym import Env
from gym.spaces import Box


class Projecitions(Env):
    """Custom Environment that follows gym interface"""
    metadata = {"render.modes": ["human"]}

    def __init__(self, X, y):
        super(Projecitions, self).__init__()

        # Pass data
        self.X = X
        self.y = y.values

        # Shuffle data
        self.SEED = np.random.randint(low=1,high=151)
        np.random.seed(self.SEED)
        np.random.shuffle(self.X)
        np.random.seed(self.SEED)
        np.random.shuffle(self.y)

        # Define action and observation space
        self.action_space = Box(low=0.0,high=500.0, shape=(1,))
        self.observation_space = Box(low=-100, high=100,
                                            shape=(10,), dtype=np.float64)

        # Initialize first step and our total steps
        self.current_step = 0
        self.total_steps = len(X)

    def step(self, action):
        observation = self.X[self.current_step]
        
        # Calculate reward  
        reward = self.__reward__(action=action,trueValue=self.y[self.current_step])
        self.score += reward 

        # Update steps
        self.current_step += 1
        self.total_steps -= 1

        # End episode if total score is too low, i.e. you lose
        # if self.score <= -12000 or self.total_steps <= 0:
        #     self.done = True
        # else:
        #     self.done = False

        # Write logic to see if episode is done
        if self.total_steps <= 0:
            self.done = True
        else:
            self.done = False

        info = {'Total Score': self.score}

        return observation, reward, self.done, info
    def reset(self):

        # Shuffle data
        self.SEED = np.random.randint(low=1,high=151)
        np.random.seed(self.SEED)
        np.random.shuffle(self.X)
        np.random.seed(self.SEED)
        np.random.shuffle(self.y)

        self.done = False
        self.score = 0.0
        self.current_step = 0
        self.total_steps = len(self.X)

        observation = self.X[self.current_step]

        return observation  # reward, done, info can't be included

    def __reward__(self, action, trueValue):
        OPTIMAL_RANGE = np.array([trueValue*1.05, trueValue*0.95])  # Projection is within +/- 10% of the acutal value
        HIGH_RANGE = np.array([trueValue*1.1, trueValue*0.9])
        MID_RANGE = np.array([trueValue*1.15, trueValue*0.85])
        LOW_RANGE = np.array([trueValue*1.25, trueValue*.75])

        if OPTIMAL_RANGE[0] > action > OPTIMAL_RANGE[1]:
            reward = 100.0
        elif HIGH_RANGE[0] > action > HIGH_RANGE[1]:
            reward = 50.0
        elif MID_RANGE[0] > action > MID_RANGE[1]:
            reward = 12.5
        elif LOW_RANGE[0] > action > LOW_RANGE[1]:
            reward = 1
        else:
            reward = -5.0

        return reward


    def render(self, mode="human"):
        pass