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
    'Neck': 12,
    'Spine1': 3,
    'L_Shoulder': 16, 'L_Elbow': 18,
    'R_Shoulder': 17, 'R_Elbow': 19,
    'L_Hip': 1,       'L_Knee': 4, 'L_Ankle': 7,
    'R_Hip': 2,       'R_Knee': 5, 'R_Ankle': 8
}

# LR reverse for deepmimic
target_joints = {
    'Neck':  [12, 13, 14, 15],
    'Spine1': [8, 9, 10, 11],
    'L_Shoulder': [39, 40, 41, 42], 'L_Elbow': [43],
    'R_Shoulder': [25, 26, 27, 28], 'R_Elbow': [30],
    'L_Hip': [16, 17, 18, 19],      'L_Knee': [20],       'L_Ankle': [21, 22, 23, 24],
    'R_Hip': [30, 31, 32, 33],      'R_Knee': [34],       'R_Ankle': [35, 36, 37, 38],
}

def to_euler_xyz(x):
    x[0], x[1], x[2] = -x[2], x[1], x[0]
    th = np.linalg.norm(x)
    x_norm = x / th
    q = quaternions.axangle2quat(x_norm, th)
    a = euler.quat2euler(q)
    return a

def to_quaternion(x):
    x[0], x[1], x[2] = -x[2], x[1], x[0]
    th = np.linalg.norm(x)
    x_norm = x / th
    q = quaternions.axangle2quat(x_norm, th)
    return q

def to_angle(x):
    x[0], x[1], x[2] = -x[2], x[1], x[0]
    th = np.linalg.norm(x)
    x_norm = x / th
    print("axis: {}, angle: {}".format(x_norm, np.rad2deg(th)))
    return 2 * np.pi - th

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

    def __call__(self, img_path, J3d=False):
        input_img = self._preprocess(img_path)
        q3d0, q3d_pred, J3d = self._model.predict(input_img)
        #joints, verts, cams, joints3d, theta = self._model.predict(input_img, get_theta=True)
        # theta SMPL angles
        num_steps = input_img.shape[0]
        x3d0 = np.zeros((num_steps, 44))
        x3dp = np.zeros((num_steps, 44))
        for i in range(num_steps):
            x3d0[i] = self.kinematicTree(q3d0[i])
            x3dp[i] = self.kinematicTree(q3d_pred[i])
        if J3d:
            return x3d0, x3dp, J3d
        else:
            return x3d0, x3dp

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
        # motions[:, 0] = 0.0625
        # motions[:, 4:8] = [1, 0,0,0]    # root rotation
        # motions[:, 8:12] = [1, 0,0,0]   # chest rotation
        # motions[:, 12:16] = [1, 0, 0, 0]  # neck rotation
        # motions[:, 16:20] = [1, 0, 0, 0] # right hip rot
        # motions[:, 20] = [1, 0, 0, 0] # right knee
        # motions[:, 21:25] = [1, 0, 0, 0] # right ankle rot
        # motions[:, 25:29] = [1, 0, 0, 0] # right shoulder rotation
        # motions[:, 30] = [1, 0, 0, 0] # right elbow
        # motions[:, 30:34] = [1, 0, 0, 0] # left hip rot
        # motions[:, 34] = [1, 0, 0, 0] # left knee
        # motions[:, 35:39] = [1, 0, 0, 0] # left ankle
        # motions[:, 39:43] = [1, 0, 0, 0] # left shoulder rot
        # motions[:, 43] = [1, 0, 0, 0] # left elbow rot


        #theta = theta[self.num_cam:(self.num_cam + self.num_theta)]
        theta = theta.reshape((-1,3))
        z = np.zeros(44)
        for joi, num in joints.items():
            print("{}:".format(joi))
            x = theta[num]
            if joi in ['L_Elbow', 'R_Elbow', 'R_Knee', 'L_Knee']:
                a = to_angle(x)
            else:
                a = to_quaternion(x)
            print(a)
            z[target_joints[joi]] = a
        return z
