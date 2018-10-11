import argparse
import logging
import sys
import time

from absl import flags
from src.tf_pose import common
import cv2
import numpy as np
import skimage.io as io
from src.util import image as img_util
from src.util import openpose as op_util
from src.util import renderer as vis_util

from src.tf_pose.get_people import get_people
import tensorflow as tf
from src.RunModel import RunModel
import src.config
import quaternion

parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--image', type=str, default='./data/coco3.png')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

args = parser.parse_args()

def get_angles(joints, cam, proc_param):
    #(0 - r ankle,
    # 1 - r knee,
    # 2 - r hip,
    # 3 - l hip,
    # 4 - l knee,
    # 5 - l ankle,
    # 6 - pelvis,
    # 7 - thorax,
    # 8 - upper neck,
    # 9 - head top,
    # 10 - r wrist,
    # 10 - r wrist,
    # 12 - r shoulder,
    # 13 - l shoulder,
    # 14 - l elbow,
    # 15 - l wrist)
    limb_parents = [1, 2, 12, 12, 3, 4, 7, 8, 12, 12, 9, 10, 13, 13, 13]

    import matplotlib.pyplot as plt
    from mpl_toolkits import mplot3d
    joints = joints[0,:14,:]

    #print(joints)

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


def main(img_path):
    sess = tf.Session()
    model = RunModel(config, sess=sess)
    kps = get_people(img_path)

    input_img, proc_param, img = preprocess_image(img_path, kps)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    theta = theta[0,3:75]
    q = quaternion.from_rotation_vector(theta[3:6])
    print(q)
    q = quaternion.from_rotation_vector(theta[6:9])
    print(q)
    get_angles(joints3d, cams[0], proc_param)
    #visualize(img, proc_param, joints[0], verts[0], cams[0])

if __name__ == "__main__":
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    img = args.image
    main(img)

