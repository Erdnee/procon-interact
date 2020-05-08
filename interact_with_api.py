import requests
import json 
import algorithms.qlearn as qlearn
admin_token = "qXf3PTcS41"
url = 'http://localhost:8081'
headers = {'Accept' : '*/*', 'Content-Type' : 'application/json'}

response = requests.post(url+'/admin/'+admin_token+'/initgame', headers = headers, data=open('init-field.json', 'rb'))
if response.status_code == 200:
    print("status code:",response.status_code,"while posting initial game data")
else:
    print("status code:",response.status_code,"while posting initial game data")

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
team1_token = token_data[0]["token"]
team2_token = token_data[1]["token"]

response = requests.post(url+'/admin/'+admin_token+'/inittoken',headers = headers, data=token_json)
if response.status_code == 200:
    print("status code:",response.status_code,"while posting initial token data")
else:
    print("status code:",response.status_code,"while posting initial token data")


response = requests.get(url+"/status")
if response.status_code == 200:
    print("status code:",response.status_code,"while fetching game data")
    status = response.json() # game status: field data, team data, agents data .,etc
else:
    print("status code:",response.status_code,"while fetching game data")

print(status["width"])

env = qlearn.Env(10,10)
env.render()