# -*- coding: utf-8 -*-

# Device setting for computation
DEVICE = '0'  # For GPU set device num which you want to use (or set 'cpu', but it's too slow)
# DEVICE = 'cpu'

# Confidence threshold for person and bag detection (bbox)
PERSON_CONF = 0.8
BAG_CONF = 0.8

# List of keypoints used in detection
KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

# Confidence threshold for keypoints
KPTS_CONF = 0.8

# IOU threshold for bounding box matching
IOU_THRESHOLD = 0.5

# XMem original config, can be changed for specific tasks
XMEM_CONFIG = {
    'top_k': 30,
    'mem_every': 5,
    'deep_update_every': -1,
    'enable_long_term': True,
    'enable_long_term_count_usage': True,
    'num_prototypes': 256,
    'min_mid_term_frames': 7,
    'max_mid_term_frames': 20,
    'max_long_term_elements': 10000,
}

# Max possible count of objects (persons, bags, backpacks) in video (increase if needed)
MAX_OBJECT_CNT = 20

# Check new objects (persons, bags, backpacks) in frame every N frames
YOLO_EVERY = 20

# Resize processed video. For better results, increase resolution
INFERENCE_SIZE = (960, 500)

# DeepSORT configuration settings
DEEPSORT_MAX_AGE = 30  # Max age for tracked objects
DEEPSORT_N_INIT = 3    # Number of frames an object needs to be observed to be considered as tracked

# Categories to be tracked
CATEGORIES = ['person', 'suitcase', 'backpack']
