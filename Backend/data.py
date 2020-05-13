from time import sleep
import requests
import json


boards_endpoint = "https://a.4cdn.org/boards.json"


boards_json = requests.get(boards_endpoint)

boards_json = boards_json.json()

print(boards_json["boards"])

print("All Boards")

for board in (boards_json["boards"]):
    print(board["title"]," ",board["pages"])