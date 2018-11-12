from absl import flags
import os
import src.config
import sys
from src.MoRecSkeletonExtractor import MoRecSkeletonExtractor
import numpy as np

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
#config.batch_size = 1

e = MoRecSkeletonExtractor(config)
img_dir = "data/youtube/1/"
z0, z_pred = e(img_dir)

np.save('results/data0.npy', z0)
np.save('results/data_pred.npy', z_pred)
#np.save('j3d.npy', z)
