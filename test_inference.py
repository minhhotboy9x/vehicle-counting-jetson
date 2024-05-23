import sys
sys.path.append('src')
from model.M_Detection import DetectionModel
from imread_from_url import imread_from_url

if __name__ == '__main__':
    model = DetectionModel("model/trt_jetson/yolov8n_relu_FP16.engine")

    # model.gen_detection(3)
    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)
    model(img)
    print('--infer')