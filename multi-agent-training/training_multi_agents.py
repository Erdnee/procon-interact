import requests
import pickle
from grid_env import Env
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch
import numpy as np
from itertools import count
import argparse
import json
import time
parser = argparse.ArgumentParser()
parser.add_argument('agentNo', type=int, help='1 for agent1, 2 for agent2')
GAMMA = 0.9
eps = np.finfo(np.float32).eps.item()
alpha = 0.05
with open('../init-field.json') as f:
    init_field = json.load(f)
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
admin_token = "qXf3PTcS41"
token = ["team1","team2"]
with open('../init-token.json') as f:
    game_token = json.load(f)
def start_game(url, headers):
    response = requests.get(url+"/admin/"+admin_token+"/startgame")
def stop_game(url, headers):
    response = requests.get(url+"/admin/"+admin_token+"/stopgame")
def pause_game(url, headers):
    response = requests.get(url+"/admin/"+admin_token+"/pausegame")
def reset_server(url, headers,field_info,token_info):
    response = requests.post(url+"/admin/"+admin_token+"/initgame",json = field_info)
    response = requests.post(url+"/admin/"+admin_token+"/inittoken",json = token_info)
    start_game(url,headers)
def fetch(url,headers):
    response = requests.get(url+"/status")
    if response.status_code == 200:
        status = response.json() # game status: field data, team data, agents data .,etc
        return status
def move(url,headers,move,agentID):
    move_json = """
        {
            "actions": [
                {
                "agentID": 0,
                "apply": 1,
                "dx": 0,
                "dy": 0,
                "turn": 0,
                "type": "move"
                }
            ]
        }
    """
    move_object = json.loads(move_json)
    move_object["actions"][0]["agentID"] = agentID
    move_object["actions"][0]["dx"] = move[1]
    move_object["actions"][0]["dy"] = move[0]
    #print(move_object)
    response = requests.post(url+"/procon/"+token[agentID-1]+"/move",json = move_object)
    #print(response.status_code," while moving ")
    return response.status_code
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



reset_server(url,headers,init_field,game_token)
status = fetch(url,headers)
env1 = Env(status,1)
env2 = Env(status,2)
policy1 = PolicyNetwork(env1.col_size * env1.row_size * 2, 9,198)
policy2 = PolicyNetwork(env2.col_size * env2.row_size * 2, 9,198)
optimizer1 = optim.SGD(policy1.parameters(), lr=1e-4)
optimizer2 = optim.SGD(policy2.parameters(), lr=1e-4)
done = False
sent_request = False
for i_episode in count(1):
    if done:
        break
    if i_episode % 300000 == 0:
        done = True
    reset_server(url,headers,init_field,game_token)
    status = fetch(url,headers)
    env1 = Env(status,1)
    env2 = Env(status,2)
    
    state1, ep_reward1 = env1.get_state(), 0
    state2, ep_reward2 = env2.get_state(), 0
    sent_request = False
    prev_turn = status["turn"]
    while status["turn"] != status["turnLimit"]:
        if prev_turn != status["turn"]:
            prev_turn = status["turn"]
            sent_request = False
        if sent_request == False:
            if(status["turn"] % 2 == 0):
                action1 = select_action(state1,policy1)
                print("move1:",env1.get_move(action1),"action1: ",action1)
                status_code = move(url,headers,env1.get_move(action1),1)
                state1, reward1, done1 = env1.step(action1)
                ep_reward1 += reward1
                policy1.rewards.append(reward1)
            else:
                action2 = select_action(state2,policy2) 
                print("move2:",env2.get_move(action2),"action2: ",action2)
                status_code = move(url,headers,env2.get_move(action2),2)
                state2, reward2, done2 = env2.step(action2)
                ep_reward2 += reward2
                policy2.rewards.append(reward2)
            if status_code == 200:
                sent_request = True
        else:
            time.sleep(0.5)
        if(i_episode % 100 == 0):
            with open("multi-agent-policy-1.pickle","wb") as file:
                pickle.dump(policy1,file)
            with open("multi-agent-policy-2.pickle","wb") as file:
                pickle.dump(policy2,file)
            print("saving policies")
    print('run: ', i_episode)
        status = fetch(url,headers)
        env1 = Env(status,1)
        env2 = Env(status,2)
        # print('running: {:d} samples\r'.format(i_episode), end = '')
    # if (i_episode % 1000 == 0):
    #     print('episode reward: ', ep_reward)

    finish_episode(policy1,optimizer1)
    finish_episode(policy2,optimizer2)