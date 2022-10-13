from stable_baselines3.common.env_checker import check_env
from environments import Projecitions
from datapreper import get_data

trainX, trainy, _, _, _, _ = get_data()

env = Projecitions(X=trainX,y=trainy)
check_env(env)