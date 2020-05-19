import argparse
parser = argparse.ArgumentParser()
parser.add_argument('teamID', metavar='teamID', type=int, nargs='+',help='insert teamID')
args = parser.parse_args()
import json 
from algorithms.policy_temp_copy import Env
from algorithms.policy_temp_copy import PolicyNetwork 
import algorithms.policy_temp_copy as reinforce
import numpy as np
import torch.nn as nn
import pickle
import requests
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
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
if teamID == 1:
    token = token_data[0]["token"]
else: 
    token = token_data[1]["token"]

#response = requests.get(url+"/status") ## fetching game status