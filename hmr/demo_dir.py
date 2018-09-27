"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px.

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np
import os

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/mpii_3dhp_ts6', 'dir to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')
flags.DEFINE_string('output_path', 'output', 'output dir')


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


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img

def p3d(joints, fn):
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
    # plt.ion()
    plt.figure(1)
    plt.clf()
    ax = plt.axes(projection='3d')
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.view_init(elev=-70, azim=-90)
    ax.scatter3D(joints[:,0],
              joints[:,1],
              joints[:,2], 'gray')

    for i in range(joints.shape[0]):
        x_pair = [joints[i, 0], joints[limb_parents[i], 0]]
        y_pair = [joints[i, 1], joints[limb_parents[i], 1]]
        z_pair = [joints[i, 2], joints[limb_parents[i], 2]]
        ax.plot(x_pair, y_pair, zs=z_pair, linewidth=3)
    #ax.axis('off')
    #plt.show()
    plt.savefig(fn)

def main(img_path, out_path, json_path=None):
    sess = tf.Session()
    model = RunModel(config, sess=sess)


    if os.path.isdir(img_path):
        print(img_path)
        img_files = [f for f in os.listdir(img_path)
                     if os.path.isfile(os.path.join(img_path, f))]

        for f in img_files:
            f_path = os.path.join(img_path, f)
            input_img, proc_param, img = preprocess_image(f_path, json_path)

            # Add batch dimension: 1 x D x D x 3
            input_img = np.expand_dims(input_img, 0)

            joints, verts, cams, joints3d, theta = model.predict(
                                        input_img, get_theta=True)

            print(joints3d)
            f_out = os.path.join(out_path, f)
            p3d(joints3d, f_out)
    #visualize(img, proc_param, joints[0], verts[0], cams[0])


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    #renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.output_path, config.json_path)
