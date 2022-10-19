import os
import sys

path = os.path.dirname(os.getcwd()) + '\\functions'
sys.path.insert(0,path)

from datapreper import *


clean_k()
engineer_data()