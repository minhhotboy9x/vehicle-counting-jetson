from flask import Blueprint, Response, request, jsonify
from model.M_Detection import DetectionModel
from config import MODEL, FRAME_WIDTH, FRAME_HEIGHT
import numpy as np


detect_bp = Blueprint('detect', __name__)

detection = DetectionModel(MODEL)

@detect_bp.route('/crop_points')
def crop_points():
    xmin = request.args.get("xmin", 0, type=int)
    ymin = request.args.get("ymin", 0, type=int)
    xmax = request.args.get("xmax", FRAME_WIDTH, type=int)
    ymax = request.args.get("ymax", FRAME_HEIGHT, type=int)
    detection.offset = np.array([[xmin, ymin], [xmax, ymax]])
    # print(detection.offset)
    return jsonify({'message': 'sent crop point successful'}), 200

@detect_bp.route('/detecting/<int:cam_id>')
def stream_cam(cam_id):
    # return Response(detection.gen_detection2(cam_id), mimetype='text/event-stream')
    return Response(detection.gen_detection(cam_id), mimetype='text/event-stream')
