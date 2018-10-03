import argparse
import logging
import sys
import time

from absl import flags
import cv2
import numpy as np
import skimage.io as io
from src.util import image as img_util
from src.util import openpose as op_util

from src.tf_pose.get_people import get_people
import tensorflow as tf
from .RunModel import RunModel
import src.config as config

def inner_angle(v0, v1, degree=True):
    angle = np.math.atan2(np.linalg.norm(np.cross(v0, v1)),np.dot(v0, v1))
    if degree:
        return np.degrees(angle)
    return angle

def normalize(v):
    return v / np.linalg.norm(v)

def normal_vector(v1, v2):
    return np.cross(v1, v2)

def populateMatrix(v1, v2, v3):
    R = np.empty((3,3))
    R[0,:] = v1
    R[1,:] = v2
    R[2,:] = v3
    return R

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

class SkeletonExtractor:
    def __init__(self, config):
        self.sess = tf.Session()
        self._model = RunModel(config, sess=self.sess)

    def __call__(self, img_path):
        kps = get_people(img_path)
        input_img, proc_param, img = preprocess_image(img_path, kps)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)
        joints, verts, cams, joints3d, theta = self._model.predict(input_img, get_theta=True)
        joints3d = joints3d[0,:14]
        self.kinematicTree(joints3d)

    def kinematicTree(self, z):
        """
        z: 3D joint coordinates 14x3
        v: vectors
        """

        # 0 right foot
        # 1 right knee
        # 2 right hip
        # 3 left hip
        # 4 left knee
        # 5 left foot
        # 6 right hand
        # 7 right elbow
        # 8 right shoulder
        # 9 left shoulder
        # 10 left elbow
        # 11 left hand
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

        x = np.zeros(25)

        z_pelvis = (z[2] + z[3]) / 2.


        # 8: abdomen z
        # 9: abdomen y
        # 10: abdomen x

        # 11: right hip x
        # 12: right hip z
        # 13: right hip y

        # 14: right knee
        x[14] = inner_angle(z[1]-z[2], z[1]- z[0])


        # 15: left hip x
        # 16: left hip z
        # 17: left hip y

        # 18: left knee
        x[18] = inner_angle(z[4]-z[3], z[4]- z[5])

        # 19: right shoulder 1
        # local y : vector pointing from right shoulder to left
        # local z : vector pointing from pelvis to thorax
        # original vector [0 0 0] -> [.16 .16 -.16]
        v_orig_l = normalize(np.array([1, -1, -1]))
        v_new_g = normalize(z[8] - z[7])

        v_y = normalize(z[8] - z[9])
        v_z = normalize(z[12] - z_pelvis)
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_l = R.transpose().dot(v_new_g)
        # rotation axis 1 = [2 -1 -1]
        a_1 = normalize(np.array([2,-1,-1]))
        x[19] = inner_angle(v_new_l.dot(a_1), v_orig_l.dot(a_1))

        # 20: right shoulder 2
        # rotation axis 2 = [0 1 1]
        a_2 = normalize(np.array([0, 1, 1]))
        x[20] = inner_angle(v_new_l.dot(a_2), v_orig_l.dot(a_2))

        # 21: right elbow
        x[21] = inner_angle(z[7]-z[6], z[7]- z[8])

        # 22: left shoulder 1
        v_orig_l = normalize(np.array([1, 1, -1]))
        v_new_g = normalize(z[10] - z[9])

        v_y = normalize(z[8] - z[9])
        v_z = normalize(z[12] - z_pelvis)
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_l = R.transpose().dot(v_new_g)

        # 23: left shoulder 2
        #x[23] =
        # 24: left elbow
        x[24] = inner_angle(z[10]-z[11], z[10]- z[9])

        print(x)
