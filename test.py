import requests
import json
import base64
from PIL import Image
from io import BytesIO

url = 'http://localhost:5001/detecting/1'

response = requests.get(url, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            info = data['boxes']
            image_data = base64.b64decode(data['img'])

            print("Received info:", info)

            # Lưu và mở ảnh
            img = Image.open(BytesIO(image_data))
            img.show()
else:
    print(f"Failed to connect, status code: {response.status_code}")
