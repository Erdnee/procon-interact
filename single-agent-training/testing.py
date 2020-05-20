import requests
import pickle
from grid_env import Env
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
def fetch():
    url = 'http://localhost:8081'
    headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
    response = requests.get(url+"/status")
    if response.status_code == 200:
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
status = fetch()
env = Env(status,2)
state = env.get_state()
load_file_name = "agent2_policy.pickle"
pickle_in = open(load_file_name,"rb")
print("loaded from: ",load_file_name)
policy = pickle.load(pickle_in)
while env.cround != env.nround:
    action = select_action(state,policy)
    state, reward, done = env.step(action)
    env.render()
    if done:
        break