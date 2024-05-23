import numpy as np
import cv2
import warnings

class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
               'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
               'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
               'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
               'scissors', 'teddy bear', 'hair drier', 'toothbrush']

# Create a list of colors for each class where each color is a tuple of 3 integer values
rng = np.random.default_rng(3)
colors = rng.uniform(0, 255, size=(len(class_names), 3))


def nms(boxes, scores, iou_threshold):
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]

    keep_boxes = []
    while sorted_indices.size > 0:
        # Pick the last box
        box_id = sorted_indices[0]
        keep_boxes.append(box_id)

        # Compute IoU of the picked box with the rest
        ious = compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

        # Remove boxes with IoU over the threshold
        keep_indices = np.where(ious < iou_threshold)[0]

        # print(keep_indices.shape, sorted_indices.shape)
        sorted_indices = sorted_indices[keep_indices + 1]

    return keep_boxes

def multiclass_nms(boxes, scores, class_ids, iou_threshold):

    unique_class_ids = np.unique(class_ids)

    keep_boxes = []
    for class_id in unique_class_ids:
        class_indices = np.where(class_ids == class_id)[0]
        class_boxes = boxes[class_indices,:]
        class_scores = scores[class_indices]

        class_keep_boxes = nms(class_boxes, class_scores, iou_threshold)
        keep_boxes.extend(class_indices[class_keep_boxes])

    return keep_boxes

def compute_iou(box, boxes):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            
            # Compute intersection area with safeguards against negative values
            intersection_area = np.maximum(1e-8, np.minimum(box[2], boxes[:, 2]) - np.maximum(box[0], boxes[:, 0])) * \
                                np.maximum(1e-8, np.minimum(box[3], boxes[:, 3]) - np.maximum(box[1], boxes[:, 1]))

            # Compute union area with handling of potential zero areas
            box_area = np.maximum(1e-8, box[2] - box[0]) * np.maximum(1e-8, box[3] - box[1])
            boxes_area = np.maximum(1e-8, (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
            union_area = box_area + boxes_area - intersection_area

            # Prevent division by zero and handle potential NaN values effectively
            iou = np.where(union_area > 1e-8, intersection_area / union_area, 1e-8)

            return iou

    except RuntimeWarning as e:
        print("RuntimeWarning encountered:", e)
        print("intersection_area:", intersection_area)
        print("box_area:", box_area)
        print("boxes_area:", boxes_area)
        print("union_area:", union_area)
        print("iou:", iou)
        return np.zeros_like(iou, 0.0)  # Return NaN for all IoU values


def xywh2xyxy(x):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
    # Convert bounding box (x, y, w, h) to bounding box (x1, y1, x2, y2)
            y = np.copy(x)
            y[..., 0] = x[..., 0] - x[..., 2] / 2
            y[..., 1] = x[..., 1] - x[..., 3] / 2
            y[..., 2] = x[..., 0] + x[..., 2] / 2
            y[..., 3] = x[..., 1] + x[..., 3] / 2
    except RuntimeWarning as e:
        print("y1: ", y[..., 0])
        print("y1: ", y[..., 1])
        print("y1: ", y[..., 2])
        print("y1: ", y[..., 3])
    return y


def draw_detections(image, boxes, scores, class_ids, mask_alpha=0.3):
    det_img = image.copy()

    img_height, img_width = image.shape[:2]
    font_size = min([img_height, img_width]) * 0.0006
    text_thickness = int(min([img_height, img_width]) * 0.001)

    det_img = draw_masks(det_img, boxes, class_ids, mask_alpha)

    # Draw bounding boxes and labels of detections
    for class_id, box, score in zip(class_ids, boxes, scores):
        color = colors[class_id]

        draw_box(det_img, box, color)

        label = class_names[class_id]
        caption = f'{label} {int(score * 100)}%'
        draw_text(det_img, caption, box, color, font_size, text_thickness)

    return det_img


def draw_box( image: np.ndarray, box: np.ndarray, color = (0, 0, 255),
             thickness = 2):
    x1, y1, x2, y2 = box.astype(int)
    return cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)


def draw_text(image: np.ndarray, text: str, box: np.ndarray, color = (0, 0, 255),
              font_size: float = 0.001, text_thickness = 2):
    x1, y1, x2, y2 = box.astype(int)
    (tw, th), _ = cv2.getTextSize(text=text, fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                  fontScale=font_size, thickness=text_thickness)
    th = int(th * 1.2)

    cv2.rectangle(image, (x1, y1),
                  (x1 + tw, y1 - th), color, -1)

    return cv2.putText(image, text, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 255, 255), text_thickness, cv2.LINE_AA)

def draw_masks(image: np.ndarray, boxes: np.ndarray, classes: np.ndarray, mask_alpha: float = 0.3):
    mask_img = image.copy()

    # Draw bounding boxes and labels of detections
    for box, class_id in zip(boxes, classes):
        color = colors[class_id]

        x1, y1, x2, y2 = box.astype(int)

        # Draw fill rectangle in mask image
        cv2.rectangle(mask_img, (x1, y1), (x2, y2), color, -1)

    return cv2.addWeighted(mask_img, mask_alpha, image, 1 - mask_alpha, 0)