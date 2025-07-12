import json

import requests

# Send a GET using the URL
base_URL = "http://localhost:8080"
r = requests.get(base_URL)
r_get = requests.get(base_URL)

# Print the status code
print(f'GET status code: {r_get.status_code}')
# Print the welcome message
print(f"GET response: {r_get.json().get('message','No message found')}")



data = {
    "age": 37,
    "workclass": "Private",
    "fnlgt": 178356,
    "education": "HS-grad",
    "education-num": 10,
    "marital-status": "Married-civ-spouse",
    "occupation": "Prof-specialty",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 0,
    "capital-loss": 0,
    "hours-per-week": 40,
    "native-country": "United-States",
}

# Send a POST using the data above
r_post = requests.post(base_URL + '/data/', json=data)

# Print the status code
print(f'POST Status Code: {r_post.status_code}')
# Print the result
print(f"POST result: {r_post.json().get('result', 'No prediction found')}")
