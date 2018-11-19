import logging
import sys
import time
import os

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

smpl_joint_names = {
    0:  'Pelvis',
    1:  'L_Hip',        4:  'L_Knee',            7:  'L_Ankle',           10: 'L_Foot',
    2:  'R_Hip',        5:  'R_Knee',            8:  'R_Ankle',           11: 'R_Foot',
    3:  'Spine1',       6:  'Spine2',            9:  'Spine3',            12: 'Neck',            15: 'Head',
    13: 'L_Collar',     16: 'L_Shoulder',       18: 'L_Elbow',            20: 'L_Wrist',         22: 'L_Hand',
    14: 'R_Collar',     17: 'R_Shoulder',       19: 'R_Elbow',            21: 'R_Wrist',         23: 'R_Hand',
}

joints = {
    'Pelvis': 0,
    'Neck': 12,
    'Chest': 3,
    'L_Shoulder': 16, 'L_Elbow': 18,
    'R_Shoulder': 17, 'R_Elbow': 19,
    'L_Hip': 1,       'L_Knee': 4, 'L_Ankle': 7,
    'R_Hip': 2,       'R_Knee': 5, 'R_Ankle': 8
}

# LR reverse for deepmimic
target_joints = {
    'Pelvis': [4, 5, 6, 7],
    'Neck':  [12, 13, 14, 15],
    'Chest': [8, 9, 10, 11],
    'L_Shoulder':  [39, 40, 41, 42], 'L_Elbow': [43],
    'R_Shoulder':  [25, 26, 27, 28], 'R_Elbow': [29],
    'L_Hip': [30, 31, 32, 33],      'L_Knee': [34],       'L_Ankle': [35, 36, 37, 38],
    'R_Hip': [16, 17, 18, 19],      'R_Knee': [20],       'R_Ankle': [21, 22, 23, 24],
}

def vec_to_quaternion(x):
    th = np.linalg.norm(x)
    x_norm = x / th
    q = quaternions.axangle2quat(x_norm, th)
    return q

def vec_to_angle(x):
    th = np.linalg.norm(x)
    x_norm = x / th
    return 2 * np.pi - th

class MoRecSkeletonExtractor:
    def __init__(self, config):
        self.sess = tf.Session()
        self._model = MotionReconstructionModel(config, sess=self.sess)
        self.num_cam = 3
        self.num_theta = 72
        self.picture_size = 224
        self.num_channels = 3

    def __call__(self, img_path, get_J3d=False):
        input_img_seq, process_params = self._preprocess(img_path)
        q3d0, q3d_pred, J3d, cams = self._model.predict(input_img_seq)
        #joints, verts, cams, joints3d, theta = self._model.predict(input_img, get_theta=True)
        # theta SMPL angles
        num_steps = input_img_seq.shape[0]
        x3d0 = np.zeros((num_steps, 44))
        x3dp = np.zeros((num_steps, 44))

        for i in range(num_steps):
            x3d0[i] = self.kinematicTree(cams[i], process_params[i], q3d0[i])
            x3dp[i] = self.kinematicTree(cams[i], process_params[i], q3d_pred[i])
        if get_J3d:
            return x3d0, x3dp, J3d
        else:
            return x3d0, x3dp

    def calcRootTranslation(self, cam, proc_param):
        img_size = proc_param['img_size']
        undo_scale = 1. / np.array(proc_param['scale'])

        cam_s = cam[0]
        cam_pos = cam[1:]
        flength = 500.
        tz = flength / (0.5 * img_size * cam_s)
        principal_pt = np.array([img_size, img_size]) / 2.
        start_pt = proc_param['start_pt'] - 0.5 * img_size
        final_principal_pt = (principal_pt + start_pt) * undo_scale
        pp_orig = final_principal_pt / (img_size*undo_scale)
        trans = np.hstack([pp_orig, tz])
        return trans

    def locate_person_and_crop(self, img_path):
        kps = get_people(img_path)
        img = io.imread(img_path)
        if img.shape[2] == 4:
            img = img[:, :, :3]

        scale, center = op_util.get_bbox_dict(kps)
        crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                                   224)
        # Normalize image to [-1, 1]
        crop = 2 * ((crop / 255.) - 0.5)
        # Add batch dimension: 1 x D x D x 3
        crop = np.expand_dims(crop, 0)

        return crop, proc_param, img

    def _preprocess(self, img_dir):
        files = [f for f in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, f))]
        files = sorted(onlyfiles,
                           key=lambda f: int(f.rsplit('.')[0].split('_')[-1]))

        N = len(files)
        X = np.zeros((N, self.picture_size, self.picture_size, 3))
        process_params = [dict() for i in range(N)]

        for i, f in enumerate(files):
            print("File: {}".format(f))
            img_path = os.path.join(img_dir, f)

            try:
                input_img, param, img = self.locate_person_and_crop(img_path)
                X[i] = input_img
                process_params[i] = param
            except:
                print('no human detected at frame {}.'.format(i))
        return X

    def kinematicTree(self, cam, proc_param, theta):
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

        z[1:4] = calcRootTranslation(cam, proc_param)
        for joi, num in joints.items():
            x = theta[num]
            # change of coordinates from SMPL to DeepMimic
            if joi in ['R_Knee', 'L_Knee']:
                q = vec_to_angle(x)
            elif joi in ['L_Elbow', 'R_Elbow']:
                q = -vec_to_angle(x)
            elif joi in ['Pelvis']:
                q = vec_to_quaternion(x)
                q = quaternions.qmult([0.7071, 0, -0.7071, 0], q)
                q = quaternions.qmult([0, 1, 0, 0], q)
            elif joi in ['L_Shoulder']:
                q = vec_to_quaternion(x)
                q = quaternions.qmult(q, [0.7071, 0, 0, 0.7071])
                q = quaternions.qmult([0.7071, 0, -0.7071, 0], q)
            elif joi in ['R_Shoulder']:
                q = vec_to_quaternion(x)
                q = quaternions.qmult(q, [0.7071, 0, 0, -0.7071])
                q = quaternions.qmult([0.7071, 0, -0.7071, 0], q)
            else:
                q = vec_to_quaternion(x)
                q = quaternions.qmult([0.7071, 0, -0.7071, 0], q)

            z[target_joints[joi]] = q
        return z
