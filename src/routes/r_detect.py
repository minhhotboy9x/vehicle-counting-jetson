from flask import Blueprint, Response, request
from model.M_Detection import DetectionModel
detect_bp = Blueprint('detect', __name__)

detection = DetectionModel('model/trt_win/yolov8n_relu_INT8.engine')