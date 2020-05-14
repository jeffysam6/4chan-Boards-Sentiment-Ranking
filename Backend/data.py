from time import sleep
import requests
import json


boards_endpoint = "https://a.4cdn.org/boards.json"


boards_json = requests.get(boards_endpoint)

boards_json = boards_json.json()

print(boards_json["boards"])

print("All Boards")

boards = []

for board in (boards_json["boards"]):
    print(board["title"]," ",board["pages"])
    boards.append(board)

for board in boards:
    for page in range(1,11):
        # print(board["title"],page)
        sleep(1)
        boards_endpoint = f"https://a.4cdn.org/{board['title']}/{page}.json"
        print(board["title"],page,boards_endpoint)
        
        # boards_json = requests.get(boards_endpoint)
        # if(str(boards_json.status_code) == '404'):
        #     break
        # print(boards_json.status_code)