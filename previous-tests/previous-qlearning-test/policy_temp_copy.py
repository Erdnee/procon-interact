import requests
import sys
import torch
import random
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pickle
# import matplotlib.pyplot as plt
from itertools import count
from datetime import datetime

GAMMA = 0.9
eps = np.finfo(np.float32).eps.item()
DEPTH = 20
alpha = 0.05
time_now = datetime.now().strftime("%m_%d_%H_%M")
file_name = "policy_temp_"+time_now+".pickle"

class Env:
    def __init__(self,status):
        random.seed()
        self.row_size = status["width"]
        self.col_size = status["height"]
        self.grid = np.array((status["tiled"])).reshape((self.row_size,self.col_size))
        self.init_grid = self.grid.copy()
        self.points = np.array((status["points"])).reshape((self.row_size,self.col_size))
        #self.nround = status["turnLimit"]
        self.nround = 60
        self.moves = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        self.action = -1
        self.cround = 0
        self.ally_teamID = status["teams"][0]["teamID"]
        self.enemy_teamID = status["teams"][1]["teamID"]
        self.ax = status["teams"][0]["agents"][0]["y"] - 1
        self.ay = status["teams"][0]["agents"][0]["x"] - 1
        self.ix = self.ax
        self.iy = self.ay
        self.done = False
    def print_status(self):
        print(self.ally_teamID)
        print(self.enemy_teamID)
    def reset(self):
        self.done = False
        self.grid = self.init_grid.copy()
        self.ax = self.ix
        self.ay = self.iy     
        self.ep_reward = 0
        self.cround = 0
        return np.concatenate((self.grid,self.points))
    
    def render(self):
        print("<=========================================>")
        print('Applied action: {:d}\n'.format(self.action))
        print("===================Grid===================")
        for i in range(0, self.row_size):
            for j in range(0, self.col_size):
                print('{:2d} '.format(int(self.grid[i][j])), end = '')
            print('');

        # print("===================points===================")
        # for i in range(0, self.row_size):
        #     for j in range(0, self.col_size):
        #         print('{:2d} '.format(int(self.points[i][j])), end = '')
        #     print('');
        # print("<=========================================>")
        input()   
    
    def step(self,n):
        self.action = n
        x, y = self.ax, self.ay
        x += self.moves[n][0]
        y += self.moves[n][1]
        self.cround += 1
        done = False

        if self.cround >= self.nround:
            done = True
        
        if x < 0 or y < 0 or x >= self.row_size or y >= self.col_size or self.grid[x][y] == self.enemy_teamID:
            reward = -0.2
            self.ep_reward += reward
            return np.concatenate((self.grid,self.points)), reward, done
        if (self.grid[x][y] != self.ally_teamID):
            reward = self.points[x][y]*0.1
        else:
            reward = -0.1

        self.grid[self.ax][self.ay] = 1
        self.grid[x][y] = 2

        self.ax, self.ay = x, y
        return np.concatenate((self.grid,self.points)), reward, done
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
def select_action(policy,state):
    state = torch.from_numpy(state).float()
    probs = policy(state)
    # exp_probs = torch.exp(probs)
    # print(exp_probs)
    action_n = np.random.choice(9, p=np.squeeze(torch.exp(probs.detach()).numpy()))
    policy.log_probs.append(probs.squeeze()[action_n])
    return action_n

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

def write_to_file(file_name,policy):
    with open(file_name,"wb") as file:
        pickle.dump(policy,file)
    file1 = open("temp_last_policy.txt","w")
    file1.write(file_name)
def test():   
    env = Env(status) 
    load_file_name = np.loadtxt('temp_last_policy.txt', dtype='str')
    pickle_in = open(str(load_file_name),"rb")
    print("loaded from: ",load_file_name)
    policy = pickle.load(pickle_in)
    running_reward = 10
    while True:
        state, ep_reward = env.reset(), 0
        for t in range(1,DEPTH):
            action = select_action(policy,state)
            state, reward, done = env.step(action)
            ep_reward += reward
            env.render()
            print(reward)
            if done:
                break
        print('episode reward: ', ep_reward)  
def train(): 
    env = Env(status)
    policy = PolicyNetwork(env.col_size * env.row_size * 2, 9,198)
    optimizer = optim.SGD(policy.parameters(), lr=1e-4)
    for i_episode in count(1):
        state, ep_reward = env.reset(), 0
        for t in range(1, DEPTH): 
            action = select_action(policy,state)
            state, reward, done = env.step(action)
            
            if(i_episode % 50000 == 0):
                write_to_file(file_name,policy)
                if i_episode % 300000 == 0:
                    env.render()
                    
            ep_reward += reward
            policy.rewards.append(reward)
            if done:
                break
            # print('running: {:d} samples\r'.format(i_episode), end = '')
        if (i_episode % 1000 == 0):
            print('run: ', i_episode)
            print('episode reward: ', ep_reward)

        finish_episode(policy,optimizer)

def main(argv):
    if argv == "test":
        test()
    else:
        train()

if __name__ == "__main__":
   main(sys.argv[1])