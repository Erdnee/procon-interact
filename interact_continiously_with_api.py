import json 
from algorithms.policy_temp_copy import Env
from algorithms.policy_temp_copy import PolicyNetwork 
import algorithms.policy_temp_copy as reinforce
import numpy as np
import torch.nn as nn
import pickle
import requests
def move(move,token,headers,url = "http://localhost:8081/procon"):
    action = '''
        {
            "agentID": 0,
            "apply": 0,
            "dx": 0,
            "dy": 0,
            "turn": 0,
            "type": "move"
        }
    '''
    response = request.post(url+"/"+token+"move",data = action)
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
token = token_data[0]["token"]
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
prev_status
status
while True:
    response = requests.get(url+"/status")
    if response.status_code == 200:
        status = response.json() # game status: field data, team data, agents data .,etc
        if status != prev_status:
            env = Env(status)

        
    prev_status = status
# load_file_name = np.loadtxt('algorithms/temp_last_policy.txt', dtype='str')
load_file_name = "policy_temp_05_09_00_22.pickle"
pickle_in = open("algorithms/"+str(load_file_name),"rb")
print("loaded from: ",load_file_name)
policy = pickle.load(pickle_in)
