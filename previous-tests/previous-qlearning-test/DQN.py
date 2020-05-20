
######################### Will delete later #########################
import requests
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}
response = requests.get(url+"/status")
if response.status_code == 200:
    print("status code:",response.status_code,"while fetching game data")
    status = response.json() # game status: field data, team data, agents data .,etc
else:
    print("status code:",response.status_code,"while fetching game data")

class Env:
    def __init__(self,status):
        self.n = status["width"]
        self.m = status["height"]
        self.grid = status["tiled"]
        self.points = status["points"]
        self.turn_limit = status["turnLimit"]
        self.move_actions = 
    def print_status(self):
        print(self.points)
        print(self.turn_limit)
env = Env(status)
env.print_status()
