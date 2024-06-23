import tensorrt as trt
import os
import cv2
import numpy as np
import time
import json
import base64 
import torch
from collections import OrderedDict, namedtuple
from config import FRAME_WIDTH, FRAME_HEIGHT, MODEL
from imread_from_url import imread_from_url
from model.utils import *

class DetectionModel:
    def __init__(self, file_engine, conf_thres=0.3, iou_thres=0.5):
        self.offset = np.array([[0, 0], [FRAME_WIDTH, FRAME_HEIGHT]]) # offset for crop image
        self.device = torch.device("cuda:0")
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres
        Binding = namedtuple("Binding", ("name", "dtype", "shape", "data", "ptr"))
        logger = trt.Logger(trt.Logger.WARNING)
        # Read file
        with open(file_engine, "rb") as f, trt.Runtime(logger) as runtime:
            meta_len = int.from_bytes(f.read(4), byteorder="little")  # read metadata length
            metadata = json.loads(f.read(meta_len).decode("utf-8"))  # read metadata
            model = runtime.deserialize_cuda_engine(f.read())  # read engine
        context = model.create_execution_context()
        bindings = OrderedDict()
        output_names = []
        fp16 = False  # default updated below
        dynamic = False
        for i in range(model.num_bindings):
            name = model.get_binding_name(i)
            dtype = trt.nptype(model.get_binding_dtype(i))
            if model.binding_is_input(i):
                if -1 in tuple(model.get_binding_shape(i)):  # dynamic
                    dynamic = True
                    context.set_binding_shape(i, tuple(model.get_profile_shape(0, i)[2]))
                if dtype == np.float16:
                    fp16 = True
            else:  # output
                output_names.append(name)
            shape = tuple(context.get_binding_shape(i))
            im = torch.from_numpy(np.empty(shape, dtype=dtype)).to(self.device)
            bindings[name] = Binding(name, dtype, shape, im, int(im.data_ptr()))
        self.fp16 = fp16
        self.bindings = bindings  # Assigning to self.bindings
        self.output_names = output_names  # Assigning to self.output_names
        self.dynamic = dynamic  # Assigning to self.dynamic
        self.context = context  # Assigning to self.context
        self.model = model  # Assigning to self.model
        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
        batch_size = bindings["images"].shape[0]  # if dynamic, this is instead max batch size
        self.get_input_details()

    def get_input_details(self):
        self.input_name = self.model.get_binding_name(0)
        self.input_shape = self.model.get_binding_shape(0)
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]
        print('input name:', self.input_name)
        print('input shape:', self.input_shape)

    def scale_and_pad_image(self, image, target_size=(640, 640)):
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
        # print(start_x, start_y, scale)
        return padded_image, scale, start_x, start_y

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize input image
        input_img, scale, start_x, start_y = self.scale_and_pad_image(input_img, target_size=(self.input_width, self.input_height))
        # input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_tensor = np.ascontiguousarray(input_tensor)
        input_tensor = torch.from_numpy(input_tensor).to(self.device)
        input_tensor = input_tensor.half() if self.fp16 else input_tensor.float()
        # Scale input pixel values to 0 to 1
        input_tensor = input_tensor / 255.0
        return input_tensor, scale, start_x, start_y
    
    def __call__(self, image):
        return self.detect_objects(image)
    
    def inference(self, im):
        if self.dynamic and im.shape != self.bindings["images"].shape:
            i = self.model.get_binding_index("images")
            self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
            self.bindings["images"] = self.bindings["images"]._replace(shape=im.shape)
            for name in self.output_names:
                i = self.model.get_binding_index(name)
                self.bindings[name].data.resize_(tuple(self.context.get_binding_shape(i)))
        s = self.bindings["images"].shape
        assert im.shape == s, f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
        self.binding_addrs["images"] = int(im.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        y = [self.bindings[x].data for x in sorted(self.output_names)] # raw relative box
        
        return y
    
    def detect_objects(self, image):
        input_tensor, scale, start_x, start_y = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs, scale, start_x, start_y)

        return self.boxes, self.scores, self.class_ids


    def process_output(self, output, scale, start_x, start_y):
        output[0] = output[0].detach().cpu().numpy()
        predictions = np.squeeze(output[0]).T
        # print(predictions.shape)
        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        
        if len(scores) == 0:
            return [], [], []

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = multiclass_nms(boxes, scores, class_ids, self.iou_threshold)

        boxes[:, [0, 2]] -= start_x
        boxes[:, [1, 3]] -= start_y
        boxes /= scale 
        
        return boxes[indices], scores[indices], class_ids[indices]
    
    def extract_boxes(self, predictions):
        # Extract boxes from predictions
        boxes = predictions[:, :4]

        # Scale boxes to original image dimensions
        boxes = self.rescale_boxes(boxes)

        # Convert boxes to xyxy format
        boxes = xywh2xyxy(boxes)

        return boxes
    
    def rescale_boxes(self, boxes):

        # Rescale boxes to original image dimensions
        # input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        # boxes = np.divide(boxes, input_shape, dtype=np.float32)
        # boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)
    
    def gen_detection(self, cam_id):
        video_path = f'./imgs/{cam_id}.mp4'
        cap = cv2.VideoCapture(video_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(video_path)
            else:
                # Perform object detection
                frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
                start_detection_time = time.time()
                boxes = []
                scores = []
                class_ids = []
                boxes, scores, class_ids = self(frame[self.offset[0, 1]: self.offset[1, 1], self.offset[0, 0]: self.offset[1, 0], :])
                end_detection_time = time.time()
                real_fps = round(1 / (end_detection_time - start_detection_time), 1)
                print(f'{os.path.basename(MODEL)} vid {cam_id} fps: ', real_fps) # log model
                if len(boxes):
                    offset = np.tile(self.offset[0], (boxes.shape[0], 2))
                    boxes += offset
                    boxes = boxes.tolist()
                    scores = scores.tolist()
                    class_ids = class_ids.tolist()

                # Convert frame to JPEG format
                frame = cv2.resize(frame, (FRAME_WIDTH // 2, FRAME_HEIGHT // 2))
                ret, buffer = cv2.imencode('.jpeg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')

                json_data = json.dumps({
                    'model': os.path.basename(MODEL),
                    'img': frame_data,
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids,
                })
                yield f"data: {json_data}\n\n"

        cap.release()

if __name__ == '__main__':
    model = DetectionModel("model/trt_jetson/yolov8n_relu_FP16.engine")
    img = cv2.imread("imgs/1.png")
    
    model(img)
    model.draw_detections(img)