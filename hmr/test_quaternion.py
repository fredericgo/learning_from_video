from absl import flags
import src.config
import sys
from src.SkeletonExtractor import SkeletonExtractor
import numpy as np

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1

e = SkeletonExtractor(config)
f = './data/coco3.png'
z, theta = e(f)
data = e.kt(theta)
np.save('results/k_tree.npy', data)
#np.save('j3d.npy', z)
