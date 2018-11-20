import os
import numpy as np
import json
import skimage.io as io

from src.tf_pose.get_people import get_people
from src.util import openpose as op_util
from src.util import image as img_util
from smpl_to_deepmimic import (smpl_to_deepmimic)
from .MotionReconstructionModel import MotionReconstructionModel
from src.util.visualizer import Visualizer

def crop_around_person(img):
    kps = get_people(img)

    if img.shape[2] == 4:
        img = img[:, :, :3]

    scale, center = op_util.get_bbox_dict(kps)
    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               224)
    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)
    # Add batch dimension: 1 x D x D x 3
    crop = np.expand_dims(crop, 0)

    return crop, proc_param


class VideoMotionProcessor(object):
    def __init__(self, config):
        self.picture_size = 224
        self._model = MotionReconstructionModel(config)
        self._visualizer = Visualizer()

    def __call__(self, img_dir, motion_path, vis_path):
        input_img_seq, process_params, img = self._preprocess(img_dir)
        q3d0, q3d, J3d, cams = self._predict(input_img_seq)
        x3d = self._convert_smpl_to_deepmimic(q3d, cams, process_params)
        self._save_motion(x3d, motion_path)
        self._save_visualization(img, J3d, process_params, joints, verts, cams, vis_path)

    def _preprocess(self, img_dir):
        files = [f for f in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, f))]
        files = sorted(files,
                           key=lambda f: int(f.rsplit('.')[0].split('_')[-1]))
        img_paths = [os.path.join(img_dir, f) for f in files]
        imgs = [io.imread(p) for p in img_paths]

        self.num_imgs = len(files)
        self.original_size = imgs[0].shape[:2]

        X = np.zeros((self.num_imgs, self.picture_size, self.picture_size, 3))
        process_params = [dict() for i in range(self.num_imgs)]

        i_succ = 0
        for i, img in enumerate(imgs):
            print("File: {}".format(i))
            try:
                input_img, param = crop_around_person(img)
                X[i] = input_img
                process_params[i] = param
            except ValueError:
                print('no human detected at frame {}, using the last successful frame.'.format(i))
                X[i] = X[i_succ]
                process_params[i] = process_params[i_succ]
            i_succ = i
        return X, process_params, imgs


    def _predict(self, img_seq):
        return self._model.predict(img_seq)

    def _convert_smpl_to_deepmimic(self, q3d, cams, process_params):
        return smpl_to_deepmimic(q3d, cams, process_params)

    def _save_visualization(self, img, J3d, proc_params, joints, verts, cams, vis_path):
        self._visualizer.plot3d(J3d, vis_path)
        self._visualizer.plot2d(imgs, proc_params, joints, verts, cams, vis_path)

    def _save_motion(self, data, filename):
        mfile = dict()
        mfile['Loop'] = 'wrap'
        data[:, 0] = 0.0333
        mfile['Frames'] = data.tolist()
        with open(filename, 'w') as f:
            json.dump(mfile, f, indent=2)

