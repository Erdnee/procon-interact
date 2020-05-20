import random
import numpy as np  
class Env:
    def __init__(self,status,agentID):
        random.seed()
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
        self.cround = status["turn"]
        self.ax = status["teams"][agentID-1]["agents"][0]["y"] - 1
        self.ay = status["teams"][agentID-1]["agents"][0]["x"] - 1
        self.grid[self.ax][self.ay] = 1
        self.done = False
    def get_state(self):
        return np.concatenate((self.grid,self.points))
    def get_move(self,action):
        return self.moves[action]
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
        self.action = n
        x, y = self.ax, self.ay
        x += self.moves[n][0]
        y += self.moves[n][1]
        done = False

        if self.cround >= self.nround:
            done = True
        
        if x < 0 or y < 0 or x >= self.row_size or y >= self.col_size or self.grid[x][y] == 2:
            reward = -0.2
            return np.concatenate((self.grid,self.points)), reward, done
        if (self.grid[x][y] != 2):
            reward = self.points[x][y]*0.1
        else:
            reward = -0.1

        self.grid[x][y] = 1
        self.grid[self.ax][self.ay] = 2
        

        self.ax, self.ay = x, y
        return np.concatenate((self.grid,self.points)), reward, done

