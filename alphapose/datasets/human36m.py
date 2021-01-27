# -----------------------------------------------------
# Copyright (c) Shanghai Jiao Tong University. All rights reserved.
# Written by Haoyi Zhu and Hao-Shu Fang
# -----------------------------------------------------

"""Halpe Human keypoint(26 points version) dataset."""
import os

import json

import numpy as np
from tkinter import _flatten

from alphapose.models.builder import DATASET
from alphapose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy

from .custom import CustomDataset


@DATASET.register_module
class Human36M(CustomDataset):
    """ Halpe_simple 26 keypoints Person Pose dataset.

    Parameters
    ----------
    train: bool, default is True
        If true, will set as training mode.
    skip_empty: bool, default is False
        Whether skip entire image if no valid label is found. Use `False` if this dataset is
        for validation to avoid COCO metric error.
    dpg: bool, default is False
        If true, will activate `dpg` for data augmentation.
    """
    CLASSES = ['person']
    EVAL_JOINTS = list(range(26))
    num_joints = 26
    CustomDataset.lower_body_ids = (11, 12, 13, 14, 15, 16, 20, 21, 22, 23, 24, 25)
    joint_pairs = [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
        [20, 21], [22, 23], [24, 25]]
    
    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items = []
        labels = []

        self._path = self._cfg['PATH']
        paths = {'COCO': self._path['COCO'],
                 'H36M': self._path['H36M'],
                 'HALPE': self._path['HALPE']}
        
        with open(self._ann_file) as anno_file:
            annot = json.load(anno_file)

        keys = list(annot.keys())
        
        img_id = 0
        
        for key in keys:
            info = annot[key]['info']
            annos = annot[key]['annotations']
            
            path = paths[info]
            
            for anno in annos:
                abs_path = os.path.join(path, anno['image_name'])
                
                if not os.path.exists(abs_path):
                    raise IOError('Image: {} not exists.'.format(abs_path))

                label = self._check_load_keypoints(anno)
                                                                 
                if not label:
                    continue
                   
                for obj in label:
                    items.append({'path': abs_path, 'id': anno['id'], 'source': None})
                    labels.append(obj)
                    img_id += 1

        return items, labels

    def _check_load_keypoints(self, obj):
        """Check and load ground-truth keypoints"""
        # check valid bboxes
        valid_objs = []
        width = obj['width']
        height = obj['height']

        if not self._skip_empty:
            # dummy invalid labels if no valid objects are found
            valid_objs.append({
                'bbox': np.array([-1, -1, 0, 0]),
                'width': width,
                'height': height,
                'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
            })

        if max(obj['keypoints']) == 0:
            return valid_objs

        # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
        xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
        # require non-zero box area
        #if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
        if (xmax-xmin)*(ymax-ymin) <= 0 or xmax <= xmin or ymax <= ymin:
            return valid_objs

        if 'num_keypoints' in obj and obj['num_keypoints'] == 0:
            return valid_objs

        # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
        joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
        for i in range(self.num_joints):
            joints_3d[i, 0, 0] = obj['keypoints'][i * 3 + 0]
            joints_3d[i, 1, 0] = obj['keypoints'][i * 3 + 1]
            # joints_3d[i, 2, 0] = 0
            if obj['keypoints'][i * 3 + 2] >= 0.35:
                visible = 1
            else:
                visible = 0
            #visible = min(1, visible)
            joints_3d[i, :2, 1] = visible
            # joints_3d[i, 2, 1] = 0

        if np.sum(joints_3d[:, 0, 1]) < 1:
            # no visible keypoint
            return valid_objs

        if self._check_centers and self._train:
            bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
            kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
            ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
            if (num_vis / 80.0 + 47 / 80.0) > ks:
                return valid_objs

        valid_objs = [{
            'bbox': (xmin, ymin, xmax, ymax),
            'width': width,
            'height': height,
            'joints_3d': joints_3d
        }]

        return valid_objs

    def _get_box_center_area(self, bbox):
        """Get bbox center"""
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """Get geometric center of all keypoints"""
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
