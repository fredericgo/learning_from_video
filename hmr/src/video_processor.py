import numpy as np
import skimage.io as io

from src.tf_pose.get_people import get_people
from src.util import openpose as op_util
from src.util import image as img_util


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

    return crop, proc_param, img


class VideoMotionProcessor(object):
	def __init__(self):
		pass

	def __call__(self, img_dir):
		input_img_seq, process_params, img = self._preprocess(img_dir)
		self._predict(input_img_seq)

	def _preprocess(self, img_dir):
        files = [f for f in os.listdir(img_dir)
                     if os.path.isfile(os.path.join(img_dir, f))]
        files = sorted(onlyfiles,
                           key=lambda f: int(f.rsplit('.')[0].split('_')[-1]))
        img_paths = [img_path = os.path.join(img_dir, f) for f in files]
        imgs = [io.imread(img_path) for p in img_paths]

        self.num_imgs = len(files)
        self.original_size = imgs[0][:2]

        X = np.zeros((self.num_imgs, self.picture_size, self.picture_size, 3))
        imgs = np.zeros((self.num_imgs, self.original_size[0], self.original_size[1], 3))

        process_params = [dict() for i in range(N)]

        i_succ = 0
        for i, img in enumerate(imgs):
            print("File: {}".format(i))
            try:
                input_img, param = crop_around_person(img_path)
                X[i] = input_img
                process_params[i] = param
            except:
                print('no human detected at frame {}.'.format(i))
                X[i] = X[i_succ]
                process_params[i] = process_params[i_succ]
            i_succ = i
        return X, process_params, imgs


	def _predict(self):
		pass


	def _visualize(self):
		return

	def _save_motion(self):
		return

