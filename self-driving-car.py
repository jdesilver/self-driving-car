# Importing necessary libraries
import matplotlib.pyplot as plt  # For plotting and visualizations
import os  # For handling directory and file operations
from PIL import Image  # For image processing tasks
import gdown  # For downloading files from Google Drive

import argparse  # For argument parsing (used in CLI-based scripts)
import numpy as np  # Numerical operations using arrays
import struct  # For working with C-style data structures
import cv2  # OpenCV for computer vision tasks
from copy import deepcopy  # For making deep copies of objects

# Set data directory path and create it if not exists
DATA_ROOT = '/content/data'
os.makedirs(DATA_ROOT, exist_ok=True)

# Downloading video files from the given URLs using wget command
!wget -O /content/data/video1.mp4 "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/6.mp4"
!wget -O /content/data/video2.mp4 "https://storage.googleapis.com/carai-for-james/video2.mp4"
!wget -O /content/data/video3.mp4 "https://storage.googleapis.com/carai-for-james/video3.mp4"
!wget -O /content/data/video4.mp4 "https://storage.googleapis.com/carai-for-james/video4.mp4"

# Download YOLO model weights
model_path = os.path.join(DATA_ROOT, 'yolo_weights.h5')
!wget -O /content/data/yolo_weights.h5 "https://storage.googleapis.com/inspirit-ai-data-bucket-1/Data/AI%20Scholars/Sessions%206%20-%2010%20(Projects)/Project%20-%20%20Object%20Detection%20(Autonomous%20Vehicles)/yolo.h5"

# Define class labels used by the YOLO model
labels = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
          "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
          "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
          "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
          "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
          "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
          "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
          "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
          "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Define class for bounding boxes that will store the coordinates and class information
class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness=None, classes=None):
        """Initialize bounding box coordinates and objectness score."""
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness  # Objectness score of the box
        self.classes = classes  # Class probability scores
        self.label = -1  # Class label (default: -1, meaning undefined)
        self.score = -1  # Confidence score (default: -1, meaning undefined)

    def get_label(self):
        """Get the class label with the highest probability."""
        if self.label == -1:
            self.label = np.argmax(self.classes)
        return self.label

    def get_score(self):
        """Get the confidence score of the bounding box."""
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        return self.score

# Helper function to calculate interval overlap between two 1D intervals
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3

# Sigmoid activation function
def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

# Function to calculate IoU (Intersection over Union) between two bounding boxes
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin
    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union

# Preprocessing function to resize and normalize the input image for the model
def preprocess_input(image_pil, net_h, net_w):
    image = np.asarray(image_pil)
    new_h, new_w, _ = image.shape
    if (float(net_w) / new_w) < (float(net_h) / new_h):
        new_h = (new_h * net_w) / new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h) / new_h
        new_h = net_h
    new_w = int(new_w)
    new_h = int(new_h)
    resized = cv2.resize(image / 255., (int(new_w), int(new_h)))

    # Create a new blank image and paste the resized image onto it
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[int((net_h - new_h) // 2):int((net_h + new_h) // 2),
              int((net_w - new_w) // 2):int((net_w + new_w) // 2), :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

# Decode the network output and return the bounding boxes
def decode_netout(netout_, obj_thresh, anchors_, image_h, image_w, net_h, net_w):
    netout_all = deepcopy(netout_)  # Deepcopy of network output
    boxes_all = []  # Store all detected boxes

    # Iterate over different scale levels
    for i in range(len(netout_all)):
        netout = netout_all[i][0]
        anchors = anchors_[i]

        grid_h, grid_w = netout.shape[:2]
        nb_box = 3  # Number of anchor boxes per grid cell
        netout = netout.reshape((grid_h, grid_w, nb_box, -1))
        nb_class = netout.shape[-1] - 5  # Number of classes

        boxes = []

        # Apply sigmoid to xy, objectness, and class scores
        netout[..., :2] = _sigmoid(netout[..., :2])
        netout[..., 4:] = _sigmoid(netout[..., 4:])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * netout[..., 5:]
        netout[..., 5:] *= netout[..., 5:] > obj_thresh  # Apply object threshold

        # Iterate through the grid and boxes
        for i in range(grid_h * grid_w):
            row = i // grid_w
            col = i % grid_w

            for b in range(nb_box):
                objectness = netout[row][col][b][4]
                classes = netout[row][col][b][5:]

                # Skip boxes with low confidence
                if (classes <= obj_thresh).all(): continue

                # Get the bounding box coordinates
                x, y, w, h = netout[row][col][b][:4]
                x = (col + x) / grid_w
                y = (row + y) / grid_h
                w = anchors[b][0] * np.exp(w) / net_w
                h = anchors[b][1] * np.exp(h) / net_h

                box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, objectness, classes)
                boxes.append(box)

        boxes_all += boxes

    # Adjust bounding box coordinates to match original image dimensions
    boxes_all = correct_yolo_boxes(boxes_all, image_h, image_w, net_h, net_w)

    return boxes_all

# Correct the bounding box coordinates based on the original image size
def correct_yolo_boxes(boxes_, image_h, image_w, net_h, net_w):
    boxes = deepcopy(boxes_)

    # Determine scale factor for resizing boxes back to original image dimensions
    if (float(net_w) / image_w) < (float(net_h) / image_h):
        new_w = net_w
        new_h = (image_h * net_w) / image_w
    else:
        new_h = net_w
        new_w = (image_w * net_h) / image_h

    # Adjust coordinates for each box
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w) / 2. / net_w, float(new_w) / net_w
        y_offset, y_scale = (net_h - new_h) / 2. / net_h, float(new_h) / net_h

        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)

    return boxes

# Apply non-max suppression to filter overlapping boxes
def do_nms(boxes_, nms_thresh, obj_thresh):
    boxes = deepcopy(boxes_)
    if len(boxes) > 0:
        num_class = len(boxes[0].classes)
    else:
        return boxes

    for c in range(num_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue

            for j in range(i + 1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0  # Suppress overlapping box

    new_boxes = []
    for box in boxes:
        label = -1
        for i in range(num_class):
            if box.classes[i] > obj_thresh:
                label = i
                box.label = label
                box.score = box.classes[i]
                new_boxes.append(box)

    return new_boxes

# Function to draw bounding boxes and labels on the image
from PIL import ImageDraw, ImageFont
import colorsys

def draw_boxes(image_, boxes, labels):
    image = image_.copy()
    image_w, image_h = image.size

    # Load font for drawing labels
    font = ImageFont.truetype(font='/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf',
                              size=np.floor(3e-2 * image_h + 0.5).astype('int32'))
    thickness = (image_w + image_h) // 300  # Box line thickness

    # Generate unique colors for each class
    hsv_tuples = [(x / len(labels), 1., 1.) for x in range(len(labels))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    np.random.seed(10101)
    np.random.shuffle(colors)
    np.random.seed(None)

    # Draw boxes and labels on the image
    for i, box in reversed(list(enumerate(boxes))):
        c = box.get_label()
        predicted_class = labels[c]
        score = box.get_score()
        top, left, bottom, right = box.ymin, box.xmin, box.ymax, box.xmax

        label = '{} {:.2f}'.format(predicted_class, score)
        draw = ImageDraw.Draw(image)
        label_size = draw.textbbox((0, 0), label, font)
        label_size = (label_size[2], label_size[3])

        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image_h, np.floor(bottom + 0.5).astype('int32'))
        right = min(image_w, np.floor(right + 0.5).astype('int32'))

        # Print box coordinates and class label
        print(label, (left, top), (right, bottom))

        # Determine position for the label text
        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # Draw rectangle for the box and label
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw

    return image

# Import TensorFlow to load YOLO model
import tensorflow as tf

# Anchors define the size of anchor boxes at different grid scales
anchors = [[[116, 90], [156, 198], [373, 326]], 
           [[30, 61], [62, 45], [59, 119]], 
           [[10, 13], [16, 30], [33, 23]]]

# Load the YOLO model
darknet = tf.keras.models.load_model(model_path, compile=False)

# Object detection and NMS thresholds
obj_thresh = 0.4  # Threshold for objectness score
nms_thresh = 0.45  # Threshold for non-max suppression

# Detect objects in a single image using YOLO model
def detect_image(image_pil, obj_thresh=0.35, nms_thresh=0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
    new_image = preprocess_input(image_pil, net_h, net_w)  # Preprocess image
    yolo_outputs = darknet.predict(new_image)  # Run YOLO model

    # Decode and apply NMS on detected boxes
    boxes = decode_netout(yolo_outputs, obj_thresh, anchors, image_pil.size[1], image_pil.size[0], net_h, net_w)
    new_boxes = do_nms(boxes, nms_thresh, obj_thresh)
    
    # Draw boxes on the image
    return draw_boxes(image_pil, new_boxes, labels)

# Detect objects in a video and save output to a new file
def detect_video(video_path, output_path, obj_thresh=0.4, nms_thresh=0.45, darknet=darknet, net_h=416, net_w=416, anchors=anchors, labels=labels):
    vid = cv2.VideoCapture(video_path)  # Open video file
    if not vid.isOpened():
        raise IOError("Couldn't open webcam or video")

    # Get video properties (e.g., FPS, size)
    video_FourCC = int(vid.get(cv2.CAP_PROP_FOURCC))
    video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')  # Set codec to MP4
    video_fps = vid.get(cv2.CAP_PROP_FPS)
    video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                  int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    # Prepare video writer to save output
    out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)

    num_frame = 0
    # Read frames from the video
    while vid.isOpened():
        ret, frame = vid.read()
        num_frame += 1
        print("=== Frame {} ===".format(num_frame))  # Frame number
        if ret:
            # Convert frame to PIL image, run detection, and convert back to OpenCV format
            image_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            new_frame = cv2.cvtColor(np.asarray(detect_image(image_pil)), cv2.COLOR_RGB2BGR)
            out.write(new_frame)  # Write the processed frame to the output video
        else:
            break

    # Release video resources
    vid.release()
    out.release()
    print("New video saved!")

# Test the function on multiple videos
video_path = '/content/data/video2.mp4'
output_path = '/content/data/video2_detected.mp4'
detect_video(video_path, output_path)

video_path = '/content/data/video3.mp4'
output_path = '/content/data/video3_detected.mp4'
detect_video(video_path, output_path)

video_path = '/content/data/video4.mp4'
output_path = '/content/data/video4_detected.mp4'
detect_video(video_path, output_path)
