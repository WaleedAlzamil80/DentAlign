import requests
import base64

image_path = "/home/waleed/Documents/3DLearning/DDS/SmileShift/dataset/LS3D-W-balanced-20-03-2017/new_dataset/7153.jpg"

api_key = "ed898d4b-8448-4460-81dd-75e06011bf4b"

# Read the image and encode it in base64
with open(image_path, 'rb') as image_file:
    image_data = image_file.read()

# Convert the image to base64
base64_image = base64.b64encode(image_data).decode('utf-8')

# URL of the Preteeth AI API endpoint
url = "https://preteeth-ai.readme.io/reference/post_1-0-analysis-smile-curve-with-underlines"

# Headers for the request
headers = {
    "accept": "application/json",
    'Content-Type': 'application/json',
    'Authorization': api_key  # Replace with your API key
}

# Data payload with base64 encoded image
data = {
    'image_base64': base64_image
}

# Making the POST request
response = requests.post(url, json=data, headers=headers)
print(response)

# Checking the response status and content
if response.status_code == 200:
    result = response.json()
    print("Smile curve analysis result:", result)
else:
    print("Error:", response.status_code, response.text)