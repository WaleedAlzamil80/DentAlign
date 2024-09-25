import requests
import base64

# Read the image and encode it in base64
with open('path_to_your_image.jpg', 'rb') as image_file:
    image_data = image_file.read()

# Convert the image to base64
base64_image = base64.b64encode(image_data).decode('utf-8')

# URL of the Preteeth AI API endpoint
url = "https://preteeth-ai.readme.io/reference/post_1-0-analysis-smile-curve-with-underlines"

# Headers for the request
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer YOUR_API_KEY'  # Replace with your API key
}

# Data payload with base64 encoded image
data = {
    'image_base64': base64_image
}

# Making the POST request
response = requests.post(url, json=data, headers=headers)

# Checking the response status and content
if response.status_code == 200:
    result = response.json()
    print("Smile curve analysis result:", result)
else:
    print("Error:", response.status_code, response.text)