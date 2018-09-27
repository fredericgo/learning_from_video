import argparse
import logging
import sys
import time
import os

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

parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--image_dir', type=str, default='./data/dance')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

args = parser.parse_args()

def visualize(img, proc_param, joints, verts, cam):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    plt.show()
    # import ipdb
    # ipdb.set_trace()

def p3d(img, joints3d, joints, verts, cam, proc_param, filename):

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
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    img_size = proc_param['img_size']
    undo_scale = 1. / np.array(proc_param['scale'])

    cam_s = cam[0]
    cam_pos = cam[1:]
    flength = 500.
    tz = flength / (0.5 * img_size * cam_s)

    pp_orig = cam_for_render[1:] / (img_size*undo_scale)
    print(pp_orig)
    trans = np.hstack([pp_orig, tz])

    joints3d = joints3d[0,:14,:]
    joints3d = joints3d + trans
    print(joints3d)

    # plt.ion()
    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim(0,2)
    ax.set_ylim(0,2)
    ax.set_zlim(0,10)
    ax.view_init(elev=-70,azim=-90)
    ax.scatter3D(joints3d[:,0],
              joints3d[:,1],
              joints3d[:,2], 'gray')

    for i in range(joints3d.shape[0]):
        x_pair = [joints3d[i, 0], joints3d[limb_parents[i], 0]]
        y_pair = [joints3d[i, 1], joints3d[limb_parents[i], 1]]
        z_pair = [joints3d[i, 2], joints3d[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
    ax.axis('off')
    #plt.show()
    plt.savefig(os.path.join("output", filename))

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


def main(img_dir):
    sess = tf.Session()
    model = RunModel(config, sess=sess)

    print(img_dir)
    onlyfiles = [f for f in os.listdir(img_dir)
                 if os.path.isfile(os.path.join(img_dir, f))]
    for file in onlyfiles:
        img_path = os.path.join(img_dir, file)
        kps = get_people(img_path)

        input_img, proc_param, img = preprocess_image(img_path, kps)
        # Add batch dimension: 1 x D x D x 3
        input_img = np.expand_dims(input_img, 0)

        joints, verts, cams, joints3d, theta = model.predict(
            input_img, get_theta=True)

        p3d(img, joints3d, joints[0], verts[0], cams[0], proc_param, file)
        #visualize(img, proc_param, joints[0], verts[0], cams[0])

if __name__ == "__main__":
    config = flags.FLAGS
    config(sys.argv)
    config.load_path = src.config.PRETRAINED_MODEL
    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    img_dir = args.image_dir
    main(img_dir)
