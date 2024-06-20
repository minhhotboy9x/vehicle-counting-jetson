import sys, os
import numpy as np
sys.path.append('src')
from model.M_Detection import DetectionModel
from imread_from_url import imread_from_url
import cv2

def scale_and_pad_image(image, target_size=(640, 640)):
    # Lấy kích thước của ảnh ban đầu
    height, width, _ = image.shape
    
    # Tính tỷ lệ scale cho chiều dài và chiều rộng
    scale = min(target_size[0] / width, target_size[1] / height)
    new_width = int(width * scale)
    new_height = int(height * scale)
    
    # Scale ảnh
    scaled_image = cv2.resize(image, (new_width, new_height))
    
    # Tạo ảnh có kích thước target_size và padding nền trắng
    padded_image = np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)
    start_x = (target_size[0] - new_width) // 2
    start_y = (target_size[1] - new_height) // 2
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = scaled_image
    # padded_image[:new_height, :new_width] = scaled_image
    return padded_image

if __name__ == '__main__':
    print(os.path.exists('model/trt_win/v8s_relu_ghostneck_ghostp5_localprune_0.4_bftrain_head_VOC.engine'))
    model = DetectionModel("model/trt_win/v8s_relu_ghostneck_ghostp5_localprune_0.4_bftrain_head_VOC.engine")
    # img = imread_from_url('https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg')
    img = cv2.imread("imgs/1.png")
    img2, _, _, _ = model.scale_and_pad_image(img)
    res = model(img)
    print(res)
    print(img.shape)
    det_img = model.draw_detections(img)
    cv2.imshow("bc", det_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()