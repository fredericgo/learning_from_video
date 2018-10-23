import argparse
import logging
import sys
import time
import os

from absl import flags
import cv2
import numpy as np
import skimage.io as io
from src.util import image as img_util
from src.util import openpose as op_util

from src.tf_pose.get_people import get_people
import tensorflow as tf
from .MotionReconstructionModel import MotionReconstructionModel
import src.config as config
import matplotlib.pyplot as plt

from transforms3d.axangles import axangle2mat
from transforms3d import quaternions, euler

joints = {
    'Spine1': 3,
    'L_Shoulder': 16, 'L_Elbow': 18,
    'R_Shoulder': 17, 'R_Elbow': 19,
    'L_Hip': 1,       'L_Knee': 4,
    'R_Hip': 2,       'R_Knee': 5
}

target_joints = {
    'Spine1': [0,1,2],
    'L_Shoulder': [6,7,8],    'L_Elbow': [11],
    'R_Shoulder': [12,13,14], 'R_Elbow': [17],
    'L_Hip': [18,19,20],      'L_Knee': [23],
    'R_Hip': [25,26,27],      'R_Knee': [30]
}

def to_euler_xyz(x):
    x[0], x[1], x[2] = x[2], x[0], x[1]
    th = np.linalg.norm(x)
    x_norm = x / th
    q = quaternions.axangle2quat(x, th)
    a = euler.quat2euler(q)
    return a

def to_angle(x):
    th = np.linalg.norm(x)
    return th

def preprocess_image(img_path, kps):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    scale, center = op_util.get_bbox_dict(kps)
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               224)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

class MoRecSkeletonExtractor:
    def __init__(self, config):
        self.sess = tf.Session()
        self._model = MotionReconstructionModel(config, sess=self.sess)
        self.num_cam = 3
        self.num_theta = 72
        self.picture_size = 224
        self.num_channels = 3

    def __call__(self, img_path):
        input_img = self._preprocess(img_path)
        return self._model.predict(input_img)
        #joints, verts, cams, joints3d, theta = self._model.predict(input_img, get_theta=True)
        # theta SMPL angles
        #return self.kinematicTree(theta[0])

    def _preprocess(self, img_dir):
        onlyfiles = [f for f in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, f))]
        onlyfiles = sorted(onlyfiles,
                           key=lambda f: int(f.rsplit('.')[0].split('_')[-1]))

        N = len(onlyfiles)
        X = np.zeros((N, self.picture_size, self.picture_size, 3))
        for i, file in enumerate(onlyfiles):
            print("File: {}".format(file))
            img_path = os.path.join(img_dir, file)
            kps = get_people(img_path)
            input_img, proc_param, img = preprocess_image(img_path, kps)
            # Add batch dimension: 1 x D x D x 3
            input_img = np.expand_dims(input_img, 0)
            X[i] = input_img
        return X


    def kinematicTree(self, theta):
        """
        z: 3D joint coordinates 14x3
        v: vectors
        """
        # 0 r foot
        # 1 r knee
        # 2 r hip
        # 3 l hip
        # 4 l knee
        # 5 l foot
        # 6 r hand
        # 7 r elbow
        # 8 r shoulder
        # 9 l shoulder
        # 10 l elbow
        # 11 l hand
        # 12 thorax
        # 13 head

        ## GYM joints
        # 1-3: torso x,y,z
        # 4-7: torso rotation
        # 8-10: abdomen (z,y,x)
        # 11-13: right hip (x,z,y)
        # 14: right knee
        # 15-17: left hip (x,z,y)
        # 18: left knee
        # 19-20: right shoulder (1,2)
        # 21: right elbow
        # 22-23: left shoulder (1,2)
        # 24: left elbow
        theta = theta[self.num_cam:(self.num_cam + self.num_theta)]
        theta = theta.reshape((-1,3))
        z = np.zeros(32)
        for joi, num in joints.items():
            print("{}:".format(joi))
            x = theta[num]
            if joi in ['L_Elbow']:
                a = -to_angle(x)
            elif joi in ['R_Elbow', 'R_Knee', 'L_Knee']:
                a = to_angle(x)
            else:
                a = to_euler_xyz(x)
            print(a)
            z[target_joints[joi]] = a
        return z
