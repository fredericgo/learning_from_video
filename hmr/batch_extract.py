from absl import flags
import os
import src.config
import sys
from src.MoRecSkeletonExtractor import MoRecSkeletonExtractor
import numpy as np
import json

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
#config.batch_size = 1

e = MoRecSkeletonExtractor(config)
img_dir = "data/youtube/1/"
z0, z_pred = e(img_dir)

mfile = dict()
mfile['Loop'] = 'wrap'
z_pred[:, 0] = 0.0625
z_pred[:, 4:8] = [1, 0, 0, 0]
mfile['Frames'] = z_pred.tolist()
 
with open('results/humanoid3d_pitch.txt') as f:
	json.dump(mfile, f, indent=2)