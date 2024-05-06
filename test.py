import requests
import json
import numpy as np
import cv2
import base64
from email.parser import BytesParser

def process_stream(url):
    start_part = b'--frame\r\nContent-Type: application/json\r\n\r\n'.decode("utf-8")
    end_part = b'endpart\r\n'.decode("utf-8")
    with requests.get(url, stream=True) as response:
        buffer = ''
        for chunk in response.iter_content(chunk_size=8192):
            chunk = chunk.decode("utf-8")
            # Accumulate the chunk to the buffer
            buffer += chunk
            # Process each part when the boundary is found
            end_index = buffer.find(end_part)
            if buffer.startswith(start_part) and end_index != -1:
                buffer = buffer[len(start_part):]
                end_index -= len(start_part)
                json_part = buffer[:end_index]
                json_part = json.loads(json_part)
                frame_data = base64.b64decode(json_part['img'])
                # Convert the image data to numpy array
                frame_np = np.frombuffer(frame_data, dtype=np.uint8)
                # Decode the numpy array as an image
                frame_img = cv2.imdecode(frame_np, cv2.IMREAD_COLOR)
                # Now you can use the frame_img for further processing
                cv2.imshow('Frame', frame_img)
                cv2.waitKey(1)  # You may need to adjust the waitKey value
                buffer = buffer[end_index + len(end_part):]


url = "http://localhost:5001/detecting/1" # replace with the actual URL
process_stream(url)