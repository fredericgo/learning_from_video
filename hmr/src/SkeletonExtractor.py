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

    def extract(self, img_path):
        kps = get_people(img_path)
