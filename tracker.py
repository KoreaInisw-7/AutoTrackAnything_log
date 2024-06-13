# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from mobile_sam import SamPredictor, sam_model_registry
from skimage import measure
from deep_sort_realtime.deepsort_tracker import DeepSort
from inference.inference_core import InferenceCore
from inference.interact.interactive_utils import (image_to_torch, index_numpy_to_one_hot_torch)
from model.network import XMem
from config import XMEM_CONFIG, PERSON_CONF, BAG_CONF, MAX_OBJECT_CNT, DEVICE, DEEPSORT_MAX_AGE, DEEPSORT_N_INIT, CATEGORIES
from transformers import CLIPProcessor, CLIPModel
from pose_estimation import Yolov8PoseModel
from torchvision.ops import nms  # 추가

class Tracker:
    def __init__(self, xmem_config, max_obj_cnt, device):
        self.device = torch.device(f'cuda:{device}' if device != 'cpu' else 'cpu')
        self.xmem_config = xmem_config
        self.max_obj_cnt = max_obj_cnt

        self.network = XMem(self.xmem_config, './saves/XMem.pth', map_location=self.device).eval().to(self.device)
        self.processor = InferenceCore(self.network, config=self.xmem_config)
        self.processor.set_all_labels(range(1, self.max_obj_cnt + 1))
        self.deepsort = DeepSort(max_age=DEEPSORT_MAX_AGE, n_init=DEEPSORT_N_INIT, nms_max_overlap=1.0, max_iou_distance=0.7, max_cosine_distance=0.2)

        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        self.categories = CATEGORIES
        self.prev_features = None
        self.prev_bboxes = None
        self.prev_ids = None

    def masks_on_im(self, masks, image):
        result = np.zeros_like(image, dtype=np.uint8)
        for mask in masks:
            color = np.random.randint(0, 256, size=3)
            colored_mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
            masked_region = colored_mask * color
            result += masked_region.astype(np.uint8)
        return result

    def create_mask_from_img(self, image, yolov8_bboxes, sam_checkpoint='./saves/sam_vit_l.pth', model_type='vit_l'):
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self.device)
        predictor = SamPredictor(sam)
        predictor.set_image(image)
        input_boxes = torch.tensor(yolov8_bboxes, device=predictor.device)

        transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])
        masks = []

        for box in transformed_boxes:
            mask, _, _ = predictor.predict_torch(point_coords=None, point_labels=None, boxes=box.unsqueeze(0), multimask_output=False)
            values, counts = torch.unique(mask, return_counts=True)
            value_count = sorted([(v.item(), c.item()) for v, c in zip(values, counts)], key=lambda x: x[1], reverse=True)
            mask[mask != 0] = value_count[0][0] if value_count[0][0] != 0 else value_count[1][0]
            masks.append(mask)

        if not masks:
            print("Warning: No masks were created")
            return np.zeros_like(image[:, :, 0])

        result = self.masks_on_im([mask.cpu().squeeze().numpy().astype(np.uint8) for mask in masks], image)
        result = result[:, :, 0]

        if len(np.unique(result)) > len(yolov8_bboxes) + 1:
            filtered_result_values = []
            mask_uniq_values, class_pixel_cnts = torch.unique(torch.tensor(result), return_counts=True)
            sorted_indices = np.argsort(class_pixel_cnts)[::-1]

            for index in sorted_indices:
                filtered_result_values.append(mask_uniq_values[index].item())
                if len(filtered_result_values) == len(yolov8_bboxes) + 1:
                    break

            for pixel_val in mask_uniq_values:
                if pixel_val.item() not in filtered_result_values:
                    result[result == pixel_val.item()] = 0

        return result

    def masks_to_boxes_with_ids(self, mask_tensor: torch.Tensor):
        unique_values = torch.unique(mask_tensor[mask_tensor != 0])
        bbox_list = []

        for unique_value in unique_values:
            binary_mask = (mask_tensor == unique_value).byte()
            nonzero_coords = torch.nonzero(binary_mask, as_tuple=False)

            if nonzero_coords.numel() > 0:
                min_x = torch.min(nonzero_coords[:, 2])
                min_y = torch.min(nonzero_coords[:, 1])
                max_x = torch.max(nonzero_coords[:, 2])
                max_y = torch.max(nonzero_coords[:, 1])

                bbox = [unique_value.item(), min_x.item(), min_y.item(), max_x.item(), max_y.item()]
                bbox_list.append(bbox)

        return bbox_list

    def predict(self, image):
        frame_torch, _ = image_to_torch(image, device=self.device)
        return self.processor.step(frame_torch)

    def add_mask(self, image, mask):
        frame_torch, _ = image_to_torch(image, device=self.device)

        # Ensure mask values are within the valid range
        mask = np.clip(mask, 0, self.max_obj_cnt)

        mask_torch = index_numpy_to_one_hot_torch(mask, self.max_obj_cnt + 1).to(self.device)
        print('Added new mask')
        return self.processor.step(frame_torch, mask_torch[1:])

    def keep_largest_connected_components(self, mask):
        mask_np = mask.squeeze().cpu().numpy()
        unique_values = np.unique(mask_np)
        unique_values = unique_values[unique_values != 0]
        new_mask = np.zeros_like(mask_np)

        for class_value in unique_values:
            binary_mask = (mask_np == class_value).astype(np.uint8)
            _, _, w, h = cv2.boundingRect(binary_mask)
            kernel = (max(1, int(w // 4)), max(1, int(h // 4)))
            binary_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_CLOSE, kernel)
            labeled_mask, num_components = measure.label(binary_mask, background=0, return_num=True)
            component_sizes = [np.sum(labeled_mask == i) for i in range(1, num_components + 1)]
            largest_component = np.argmax(component_sizes) + 1
            new_mask[labeled_mask == largest_component] = class_value

        new_mask = torch.from_numpy(new_mask).unsqueeze(0)
        return new_mask

    def get_filtered_bboxes_by_confidence(self, frame, yolo_model):
        results = yolo_model.detect(frame)

        filtered_bboxes = []
        scores = []
        for bbox in results.xyxy[0]:
            label = results.names[int(bbox[5])]
            if label in self.categories:
                conf = bbox[4]
                if (label == 'person' and conf >= PERSON_CONF) or (label in ['suitcase', 'backpack'] and conf >= BAG_CONF):
                    x1, y1, x2, y2 = map(int, bbox[:4])
                    filtered_bboxes.append([x1, y1, x2, y2])
                    scores.append(conf)

        # Apply NMS
        bboxes_tensor = torch.tensor(filtered_bboxes, dtype=torch.float32, device=self.device)
        scores_tensor = torch.tensor(scores, dtype=torch.float32, device=self.device)
        keep_indices = nms(bboxes_tensor, scores_tensor, 0.5)
        filtered_bboxes = [filtered_bboxes[i] for i in keep_indices]

        return filtered_bboxes

    def match_objects(self, current_features, current_bboxes):
        if self.prev_features is None:
            self.prev_features = current_features
            self.prev_bboxes = current_bboxes
            self.prev_ids = list(range(len(current_bboxes)))
            return self.prev_ids

        matched_ids = []
        for cur_feat, cur_bbox in zip(current_features, current_bboxes):
            best_iou = 0
            best_id = -1
            for prev_feat, prev_bbox, prev_id in zip(self.prev_features, self.prev_bboxes, self.prev_ids):
                iou = self.calculate_iou(cur_bbox, prev_bbox)
                similarity = torch.nn.functional.cosine_similarity(cur_feat, prev_feat, dim=0).item()
                if iou > 0.5 and similarity > 0.7 and iou + similarity > best_iou:
                    best_iou = iou + similarity
                    best_id = prev_id
            if best_id == -1:
                best_id = max(self.prev_ids) + 1
            matched_ids.append(best_id)

        self.prev_features = current_features
        self.prev_bboxes = current_bboxes
        self.prev_ids = matched_ids
        return matched_ids

    def calculate_iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxB[3], boxB[3])

        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def get_features_from_clip(self, image, bboxes):
        crops = [image[y1:y2, x1:x2] for x1, y1, x2, y2 in bboxes]
        inputs = self.clip_processor(images=crops, return_tensors="pt", padding=True).to(self.device)
        features = self.clip_model.get_image_features(**inputs)
        return features
