import argparse
import logging
import sys
import time

from . import common
import cv2
import numpy as np
from .estimator import TfPoseEstimator
from .networks import get_graph_path, model_wh

parser = argparse.ArgumentParser(description='tf-pose-estimation run')
parser.add_argument('--image', type=str, default='./data/coco1.png')
parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin')

parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')
parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')


args = parser.parse_args()


def get_people(img_path):
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368))

    image = common.read_imgfile(img_path, None, None)
    wo, ho = image.shape[0], image.shape[1]

    if image is None:
        logger.error('Image can not be read, path=%s' % args.image)
        sys.exit(-1)

    humans = e.inference(image, upsample_size=args.resize_out_ratio)

    output = []
    for h in humans:
        keypoints = np.zeros((18,3))
        for k, bp in h.body_parts.items():
            keypoints[k, :] = [bp.x*ho,  bp.y*wo, bp.score]
        output.append(keypoints)
    return output
