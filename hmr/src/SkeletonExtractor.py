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
from src.ik import (solve_l_hip_angles, solve_r_hip_angles,
                    solve_l_shoulder_angles, solve_r_shoulder_angles,
                    solve_r_elbow_angles, solve_l_elbow_angles,
                    solve_l_knee_angles, solve_r_knee_angles)
import matplotlib.pyplot as plt
import quaternion

from transforms3d.axangles import axangle2mat

def inner_angle(v0, v1, degree=True):
    angle = np.math.atan2(np.linalg.norm(np.cross(v0, v1)),np.dot(v0, v1))
    if degree:
        return np.degrees(angle)
    return angle

def ccw_angle(v0, v1, a, degree=True):
    angle = np.math.atan2(a.dot(np.cross(v0, v1)),np.dot(v0, v1))
    if degree:
        return np.degrees(angle)
    return angle

def normalize(v):
    return v / np.linalg.norm(v)

def normal_vector(v1, v2):
    return np.cross(v1, v2)


def populateMatrix(v1, v2, v3):
    R = np.empty((3,3))
    R[:,0] = v1
    R[:,1] = v2
    R[:,2] = v3
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

        R = axangle2mat([1,0,0], np.deg2rad(90))
        joints3d = joints3d.dot(R)
        joints3d = joints3d[0,:14]
        return joints3d, theta[0,3:75]
        #self.kinematicTree(joints3d)

    def debug_rhip(self, z):
        x = np.zeros(25)
        z_pelvis = (z[2] + z[3]) / 2.
        v_orig_l = normalize(np.array([0, 0.01, -.34]))
        v_new_g = normalize(z[1] - z[2])
        v_y = normalize(z[3] - z[2])
        v_z = normalize(z[12] - z_pelvis)
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_l = R.transpose().dot(v_new_g)

        def zVec(v):
            return [0, 0, 0] + v.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*zVec(v_x),color='red')
        ax.quiver(*zVec(v_y),color='green')
        ax.quiver(*zVec(v_z),color='blue')
        ax.scatter(*z[2])
        ax.scatter(*z[3])
        ax.quiver(*zVec(v_new_g),color='yellow')

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.show()

    def debug_rotation(self, z):
        x = np.zeros(25)
        v_y = normalize(z[8] - z[9])
        v_z = normalize(z[13] - z[12])
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        print(R)
        q = quaternion.from_rotation_matrix(R)

        def zVec(v):
            return [0, 0, 0] + v.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*zVec(v_x),color='red')
        ax.quiver(*zVec(v_y),color='green')
        ax.quiver(*zVec(v_z),color='blue')
        ax.scatter(*z[9])
        ax.scatter(*z[8])
        #ax.quiver(*zVec(v_new_g),color='yellow')

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.show()

    def debug_rknee(self, z):
        x = np.zeros(25)
        z_pelvis = (z[2] + z[3]) / 2.
        v_orig_l = normalize(np.array([0, 0, -0.3]))
        v_x = normalize(z[1] - z[2])
        v_z = normalize([0, -v_x[2], v_x[1]])
        v_y = normalize(normal_vector(v_z, v_x))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_g = normalize(z[0] - z[1])
        v_new_l = R.transpose().dot(v_new_g)

        def zVec(v):
            return [0, 0, 0] + v.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*zVec(v_x),color='red')
        ax.quiver(*zVec(v_y),color='green')
        ax.quiver(*zVec(v_z),color='blue')
        ax.scatter(*z[2])
        ax.scatter(*z[3])
        ax.quiver(*zVec(v_new_l),color='yellow')

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.show()

    def debug_rshoulder(self, z):
        x = np.zeros(25)

        z_pelvis = (z[2] + z[3]) / 2.
        v_orig_l = normalize(np.array([0, -1, -1]))
        v_y = normalize(z[9] - z[8])
        v_z = normalize(z[13] - z[12])
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)

        v_new_g = normalize(z[7] - z[8])
        v_new_l = R.transpose().dot(v_new_g)

        def zVec(v):
            return [0, 0, 0] + v.tolist()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.quiver(*zVec(v_x),color='red')
        ax.quiver(*zVec(v_y),color='green')
        ax.quiver(*zVec(v_z),color='blue')
        ax.quiver(*zVec(v_orig_l),color='brown')
        ax.quiver(*zVec(v_new_l),color='yellow')

        ax.set_xlim(-1,1)
        ax.set_ylim(-1,1)
        ax.set_zlim(-1,1)
        plt.show()

    def kinematicTree(self, z):
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

        x = np.zeros(25)
        z_pelvis = (z[2] + z[3]) / 2.

        # 4-7: torso rotation
        v_y = normalize(z[9] - z[8])
        v_z = normalize(z[13] - z[12])
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        q = quaternion.from_rotation_matrix(R)
        x[4:8] = q.components

        # 8: abdomen z
        # 9: abdomen y
        # 10: abdomen x

        # 11: right hip x
        # 12: right hip z
        # 13: right hip y
        v_new_g = normalize(z[1] - z[2])
        v_y = normalize(z[2] - z[3])
        v_z = normalize(z[12] - z_pelvis)
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_l = R.transpose().dot(v_new_g)
        a = solve_r_hip_angles(v_new_l)
        x[11:14] = a

        # 14: r knee
        v_leg = normalize(z[1] - z[2])
        v_thigh = normalize(z[0] - z[1])
        a = inner_angle(v_leg, v_thigh)
        x[14] = -a

        # 15: l hip x
        # 16: l hip z
        # 17: l hip y
        # -1 0 0
        v_y = normalize(z[3] - z[2])
        v_z = normalize(z[12] - z_pelvis)
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)
        v_new_g = normalize(z[4] - z[3])
        v_new_l = R.transpose().dot(v_new_g)
        a = solve_l_hip_angles(v_new_l)
        x[15:18] = a

        # 18: left knee
        v_leg = normalize(z[4] - z[3])
        v_thigh = normalize(z[5] - z[4])
        a = inner_angle(v_leg, v_thigh)
        x[18] = -a

        # 19: right shoulder 1
        # 20: right shoulder 2
        # local y : vector pointing from right shoulder to left
        # local z : vector pointing from pelvis to thorax
        v_y = normalize(z[9] - z[8])
        v_z = normalize(z[13] - z[12])
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)

        v_new_g = normalize(z[7] - z[8])
        v_new_l = R.transpose().dot(v_new_g)

        a = solve_r_shoulder_angles(v_new_l)
        x[19] = a[0]
        x[20] = a[1]

        # 21: right elbow
        v_lowarm = normalize(z[6] - z[7])
        v_upperarm = normalize(z[7] - z[8])
        a = inner_angle(v_upperarm, v_lowarm)
        x[21] = a - 70

        # 22: left shoulder 1
        # 23: left shoulder 2
        # original vector [0 0 0] -> [.16 .16 -.16]
        v_y = normalize(z[9] - z[8])
        v_z = normalize(z[13] - z[12])
        v_x = normalize(normal_vector(v_y, v_z))
        R = populateMatrix(v_x, v_y, v_z)

        v_new_g = normalize(z[10] - z[9])
        v_new_l = normalize(R.transpose().dot(v_new_g))
        a = solve_l_shoulder_angles(v_new_l)
        x[22] = a[0]
        x[23] = a[1]

        # 24: left elbow

        v_lowarm = normalize(z[10] - z[9])
        v_upperarm = normalize(z[11] - z[10])
        a = inner_angle(v_upperarm, v_lowarm)
        x[24] = a - 70

        return x[1:]

    def kt(self, theta):
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
        # 8-11: abdomen (z,y,x)
        # 12-15: right hip (x,z,y)
        # 16-19: right knee
        # 20-23: left hip (x,z,y)
        # 24-27: left knee
        # 28-29: right shoulder (1,2)
        # 30: right elbow
        # 31-32: left shoulder (1,2)
        # 33: left elbow

        x = np.zeros(32)
        # root rotation
        q = quaternion.from_rotation_vector(theta[0:3])
        q2 = quaternion.from_rotation_vector([np.pi,0,0])*q
        q = q2.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[4:8] = q

        # abdomen
        q = quaternion.from_rotation_vector(theta[9:12])
        q = q.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[8:12] = q

        #l hip
        q = quaternion.from_rotation_vector(theta[3:6])
        q = q.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[20:24] = q

        # r hip
        q = quaternion.from_rotation_vector(theta[6:9])
        q = q.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[12:16] = q

        # l knee
        q = quaternion.from_rotation_vector(theta[12:15])
        q = q.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[24:28] = q

        # r knee
        q = quaternion.from_rotation_vector(theta[15:18])
        q = q.components
        q[1], q[2], q[3] = q[3], q[1], q[2]
        x[16:20] = q
        return x[1:]
