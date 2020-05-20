import requests
import pickle
from grid_env import Env
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from itertools import count
GAMMA = 0.9
eps = np.finfo(np.float32).eps.item()
alpha = 0.05
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
def fetch(url,headers):
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
def finish_episode(policy,optimizer):
    R = 0
    policy_loss = []
    Gt = []
    for r in policy.rewards[::-1]:
        R = r + GAMMA * R
        Gt.insert(0, R)
    Gt = torch.tensor(Gt)

    for log_prob, R in zip(policy.log_probs, Gt):
        policy_loss.append(-log_prob * R)

    optimizer.zero_grad()
    policy_loss = torch.stack(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    
    del policy.rewards[:]
    del policy.log_probs[:]
status = fetch()
env = Env(status,2)

policy = PolicyNetwork(env.col_size * env.row_size * 2, 9,198)
optimizer = optim.SGD(policy.parameters(), lr=1e-4)
for i_episode in count(1):
    env = Env(status,2)
    state, ep_reward = env.get_state(), 0
    for t in range(1, 15): 
        action = select_action(state,policy)
        state, reward, done = env.step(action)
        
        if(i_episode % 50000 == 0):
            with open("agent2_policy.pickle","wb") as file:
                pickle.dump(policy,file)
            if i_episode % 300000 == 0:
                env.render()
        print('run: ', i_episode)
        ep_reward += reward
        policy.rewards.append(reward)
        if done:
            break
        # print('running: {:d} samples\r'.format(i_episode), end = '')
    if (i_episode % 1000 == 0):
        print('episode reward: ', ep_reward)

    finish_episode(policy,optimizer)