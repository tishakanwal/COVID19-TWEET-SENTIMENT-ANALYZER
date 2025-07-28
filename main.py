import requests
import os

def fetchAndSaveToFile(url , path):
  r = requests.get(url)
  # Create the directory if it doesn't exist
  directory = os.path.dirname(path)
  if not os.path.exists(directory):
    os.makedirs(directory)
  with open(path, "w") as f:
    f.write(r.text)

url = "https://www.thehindu.com/"

r = requests.get(url)

fetchAndSaveToFile(url , "data/times.html")
