import tensorrt as trt
# import pycuda.autoinit  # Needed for managing CUDA context
import cv2
import numpy as np
# import torch
import json
import base64 
import torch
from collections import OrderedDict, namedtuple
# from config import FRAME_WIDTH, FRAME_HEIGHT
from imread_from_url import imread_from_url
from model.utils import *

class DetectionModel:
    def __init__(self, file_engine, conf_thres=0.7, iou_thres=0.5):
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

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize input image
        input_img = cv2.resize(input_img, (self.input_width, self.input_height))
        
        # Scale input pixel values to 0 to 1
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_tensor = np.ascontiguousarray(input_tensor)
        input_tensor = torch.from_numpy(input_tensor).to(self.device)
        return input_tensor
    
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
        input_tensor = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs)

        return self.boxes, self.scores, self.class_ids


    def process_output(self, output):
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
        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])
        return boxes

    def draw_detections(self, image, draw_scores=True, mask_alpha=0.4):

        return draw_detections(image, self.boxes, self.scores,
                               self.class_ids, mask_alpha)
    
    def gen_detection(self, cam_id):
        video_path = f'./imgs/{cam_id}.mp4'
        cap = cv2.VideoCapture(video_path)
        boundary = "--frame"

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap = cv2.VideoCapture(video_path)
            else:
                # Perform object detection
                frame = cv2.resize(frame, (self.input_width, self.input_height))
                boxes, scores, class_ids = self(frame) # Bạn cần định nghĩa hàm self(frame) để thực hiện phát hiện đối tượng
                if len(boxes):
                    boxes = boxes.tolist()
                    scores = scores.tolist()
                    class_ids = class_ids.tolist()

                # Convert frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_data = base64.b64encode(buffer).decode('utf-8')
                
                json_data = json.dumps({
                    'img': frame_data,
                    'boxes': boxes,
                    'scores': scores,
                    'class_ids': class_ids,
                })
                # print(len(json_data.encode('utf-8')))
                # print(json_data.encode('utf-8'))
                yield (b'--frame\r\n'
                    b'Content-Type: application/json\r\n\r\n' + json_data.encode('utf-8') + b'endpart' + b'\r\n')
        cap.release()
                

if __name__ == '__main__':
    model = DetectionModel("model/trt_win/yolov8n_win.engine")
    img_url = "https://live.staticflickr.com/13/19041780_d6fd803de0_3k.jpg"
    img = imread_from_url(img_url)
    
    model.gen_detection(1)