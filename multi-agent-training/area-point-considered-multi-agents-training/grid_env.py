import random
import numpy as np  
import requests
import json
url = 'http://localhost:8081'
token = ["team1","team2"]
def fetch(url):
    response = requests.get(url+"/status")
    if response.status_code == 200:
        status = response.json() # game status: field data, team data, agents data .,etc
        return status
status = fetch(url)
def move(url,move,agentID):
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
class Env:
    def __init__(self,agentID):
        self.row_size = status["width"]
        self.col_size = status["height"]
        self.grid = np.array((status["tiled"])).reshape((self.row_size,self.col_size))
        for i in range(0,self.row_size):
            for j in range(0,self.col_size):
                if self.grid[i][j] == 1:
                    self.grid[i][j] = 2
        self.points = np.array((status["points"])).reshape((self.row_size,self.col_size))
        self.nround = status["turnLimit"]
        self.moves = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
        self.action = -1
        self.agentID = agentID
        self.current_tile_point = status["teams"][self.agentID-1]["tilePoint"]
        self.current_area_point = status["teams"][self.agentID-1]["areaPoint"]
        self.cround = status["turn"]
        self.ax = status["teams"][agentID-1]["agents"][0]["y"] - 1
        self.ay = status["teams"][agentID-1]["agents"][0]["x"] - 1
        self.grid[self.ax][self.ay] = 1
        self.done = False
    def get_state(self):
        return np.concatenate((self.grid,self.points))
    def get_move(self,action):
        return self.moves[action]
    def reset(self):
        status = fetch(url)
        self.cround = status["turn"]
        self.current_tile_point = status["teams"][self.agentID-1]["tilePoint"]      
        self.current_area_point = status["teams"][self.agentID-1]["areaPoint"]
        self.ax = status["teams"][self.agentID-1]["agents"][0]["y"] - 1
        self.ay = status["teams"][self.agentID-1]["agents"][0]["x"] - 1
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
    
    def step(self,n):
        move(url,self.moves[n],self.agentID)
        self.reset()
        if self.cround >= self.nround:
            self.done = True
        #print((self.current_tile_point+self.current_area_point)*0.1)
        return np.concatenate((self.grid,self.points)), 0.1*(self.current_tile_point+self.current_area_point), self.done