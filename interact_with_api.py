import json 
from algorithms.policy_temp_copy import Env
from algorithms.policy_temp_copy import PolicyNetwork 
import algorithms.policy_temp_copy as reinforce
import numpy as np
import torch.nn as nn
import pickle
import requests
import asyncio

token_json = '''
    [
        {
            "teamID": 1,
            "token": "team1"
        },
        {
            "teamID": 2,
            "token": "team2"
        }
    ]
'''
token_data = json.loads(token_json)
team1_token = token_data[0]["token"]
team2_token = token_data[1]["token"]
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
response3 = requests.get(url+"/status")
if response3.status_code == 200:
    
    status = response3.json() # game status: field data, team data, agents data .,etc
#    print("status code:",response3.status_code,"while fetching game data")
env = Env(status)
# load_file_name = np.loadtxt('algorithms/temp_last_policy.txt', dtype='str')
load_file_name = "policy_temp_05_09_00_22.pickle"
pickle_in = open("algorithms/"+str(load_file_name),"rb")
print("loaded from: ",load_file_name)
policy = pickle.load(pickle_in)
state = env.reset()
while env.cround != env.nround:
    action = reinforce.select_action(policy,state)
    state, reward, done = env.step(action)
    env.render()
    if done:
        break