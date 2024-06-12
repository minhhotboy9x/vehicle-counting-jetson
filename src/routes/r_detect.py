from flask import Blueprint, Response, request
from model.M_Detection import DetectionModel
from config import MODEL
detect_bp = Blueprint('detect', __name__)



detect_bp = Blueprint('detect', __name__)

detection = DetectionModel(MODEL)

@detect_bp.route('/detecting/<int:cam_id>')
def stream_cam(cam_id):
    # return Response(detection.gen_detection2(cam_id), mimetype='text/event-stream')
    return Response(detection.gen_detection(cam_id), mimetype='text/event-stream')
