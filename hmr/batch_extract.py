from absl import flags
import os
import src.config
import sys
from src.MoRecSkeletonExtractor import MoRecSkeletonExtractor
import numpy as np
import json
import tempfile
from src.util.visualizer import Visualizer

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
#config.batch_size = 1


e = MoRecSkeletonExtractor(config)
img_dir = "data/youtube/skating/"
z0, z_pred, j3d = e(img_dir, get_J3d=True)

mfile = dict()
mfile['Loop'] = 'wrap'
z_pred[:, 0] = 0.0333
mfile['Frames'] = z_pred.tolist()
 
with open('/home/fredericgo/DeepMimic/data/motions/humanoid3d_skating.txt', 'w') as f:
	json.dump(mfile, f, indent=2)

Visualizer().plot(j3d)
