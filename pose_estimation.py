import os
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

class Yolov8PoseModel:
    def __init__(self, device: str, person_conf):
        self.device = f"cuda:{device}" if isinstance(device, int) or device.isdigit() else device
        self.person_conf = person_conf
        self.model = DetrForObjectDetection.from_pretrained('facebook/detr-resnet-50')

        # Load only the model state dict, ignoring other keys
        checkpoint = torch.load('K:/AutoTrackAnything/saves/checkpoint.pth', map_location=self.device)
        print("Checkpoint keys:", checkpoint.keys())  # Checkpoint keys 출력
        state_dict = {k: v for k, v in checkpoint['model'].items() if k.startswith('model.')}
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.processor = DetrImageProcessor.from_pretrained('facebook/detr-resnet-50')

        # Print all class labels
        print(f"Label mapping: {self.model.config.id2label}")

        # Define the person and handbag categories based on available labels
        self.person_categories = ['person', 'suitcase', 'backpack', 'handbag']

    def run_inference(self, image):
        image = Image.fromarray(image)
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]], device=self.device)
        results = self.processor.post_process_object_detection(outputs, target_sizes=target_sizes)[0]
        return results

    def get_filtered_bboxes_by_confidence(self, image):
        results = self.run_inference(image)

        print(f"Inference results: {results}")  # 디버깅 출력

        person_bboxes = []
        labels = []
        scores = []

        for score, label, box in zip(results['scores'], results['labels'], results['boxes']):
            label_name = self.model.config.id2label[label.item()]
            print(f"Label: {label_name}, Score: {score}")  # 라벨과 스코어 확인
            if score > self.person_conf and label_name in self.person_categories:
                person_bboxes.append(box.int().tolist())
                labels.append(label_name)
                scores.append(score.item())

        # Apply custom NMS to remove duplicate boxes
        person_bboxes, labels = self.apply_custom_nms(person_bboxes, labels, scores)

        print(f"Filtered bboxes: {person_bboxes}")  # 디버깅 출력

        self.visualize_inference(np.array(image), person_bboxes, labels)

        return person_bboxes

    def apply_custom_nms(self, bboxes, labels, scores, iou_threshold=0.5):
        if len(bboxes) == 0:
            return [], []

        indices = list(range(len(bboxes)))
        bboxes = np.array(bboxes)
        scores = np.array(scores)
        labels = np.array(labels)

        x1 = bboxes[:, 0]
        y1 = bboxes[:, 1]
        x2 = bboxes[:, 2]
        y2 = bboxes[:, 3]

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        unique_bboxes = bboxes[keep].tolist()
        unique_labels = labels[keep].tolist()

        return unique_bboxes, unique_labels

    def visualize_inference(self, image, bboxes, labels):
        fig, ax = plt.subplots(1, figsize=(12, 9))
        ax.imshow(image)

        for bbox, label in zip(bboxes, labels):
            x1, y1, x2, y2 = bbox
            width, height = x2 - x1, y2 - y1
            rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, label, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.5))

        plt.show()

    def get_filtered_bboxes_by_size(self, bboxes, image, percentage=10):
        image_size = image.shape[:2]
        min_bbox_width = image_size[1] * (percentage/100)  # width
        min_bbox_height = image_size[0] * (percentage/100)  # height

        filtered_bboxes = []
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            bbox_width = x2 - x1
            bbox_height = y2 - y1
            if bbox_width >= min_bbox_width and bbox_height >= min_bbox_height:
                filtered_bboxes.append(bbox)

        print(f"Filtered bboxes by size: {filtered_bboxes}")  # 디버깅 출력

        return filtered_bboxes
