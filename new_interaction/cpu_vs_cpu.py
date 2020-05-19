import requests
import time
import pickle
from grid_env import Env
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
import json
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('agentID', type=int, help='1 for agent1, 2 for agent2')
args = parser.parse_args()
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
token = ["team1","team2"]
def fetch(url,headers):
    response = requests.get(url+"/status")
    if response.status_code == 200:
        print("fetching")
        status = response.json() # game status: field data, team data, agents data .,etc
        return status
def select_action(state,policy):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    # exp_probs = torch.exp(probs)
    # print(exp_probs)
    action_n = np.random.choice(9, p=np.squeeze(torch.exp(probs.detach()).numpy()))
    policy.log_probs.append(probs.squeeze()[action_n])
    return action_n
def move(url,headers,move,turn):
    move_json = """
        {
            "actions": [
                {
                "agentID": 0,
                "apply": 1,
                "dx": 0,
                "dy": 0,
                "type": "move"
                }
            ]
        }
    """
    move_object = json.loads(move_json)
    move_object["actions"][0]["agentID"] = args.agentID
    move_object["actions"][0]["dx"] = move[1]
    move_object["actions"][0]["dy"] = move[0]
    print(move_object)
    response = requests.post(url+"/procon/"+token[args.agentID-1]+"/move",json = move_object)
    print(token[args.agentID-1])
    print(response.status_code," while moving ")
class PolicyNetwork(nn.Module):
    def __init__(self, in_features, num_actions, hidden_size, learning_rate=1e-2):
        super(PolicyNetwork, self).__init__()

        self.num_actions = num_actions
        self.linear1 = nn.Linear(in_features, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.logsoftmax = nn.LogSoftmax(dim=0)
        self.rewards = []
        self.log_probs = []
   
    def forward(self, state):
        x = F.relu(self.linear1(state.view(-1)))
        # print("x = linear1 * state",x)
        x = self.linear2(x)
        # print("x = linear2 * x",x)
        x = self.logsoftmax(x)
        # print("x = logsoftmax(x)",x)
        return x 
status = fetch(url,headers)
env = Env(status,args.agentID)
state = env.get_state()
#load_file_name = "policy_temp_05_09_00_22.pickle"
pickle_in = open("agent2_policy.pickle","rb")
print("loaded from: ","agent2_policy.pickle")
policy = pickle.load(pickle_in)
while env.cround != env.nround or status["time"] != 0:
    if env.cround % 2 == args.agentID-1:
        action = select_action(state,policy)
        state, reward, done = env.step(action)
        print("move:",env.get_move(action),"action: ",action)
        move(url,headers,env.get_move(action),status["turn"])
        env.render()
    while env.cround == status["turn"]:
        time.sleep(5)
        print("sleeping 5 seconds")
        status = fetch(url,headers)
    env = Env(status,args.agentID)
    state = env.get_state()
    env.render()
print("round ended in ",env.cround)