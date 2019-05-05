# send requests with the features to the server and receive the results

import requests


url = 'http://localhost:5000/api'
r = requests.post(url,json={'exp':1.8,})

print(r.json())
