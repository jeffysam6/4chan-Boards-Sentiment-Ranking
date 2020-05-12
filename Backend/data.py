from time import sleep
import requests


boards_endpoint = "https://a.4cdn.org/boards.json"


boards_json = requests.get(boards_endpoint)

print(boards_json.json())