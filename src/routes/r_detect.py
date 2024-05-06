from flask import Blueprint, Response, request
from model.M_Detection import DetectionModel
detect_bp = Blueprint('detect', __name__)



detect_bp = Blueprint('detect', __name__)

detection = DetectionModel('model/trt_win/yolov8n_win.engine')

@detect_bp.route('/detecting/<int:cam_id>')
def stream_cam(cam_id):
    return Response(detection.gen_detection(cam_id), mimetype='multipart/x-mixed-replace; boundary=frame')