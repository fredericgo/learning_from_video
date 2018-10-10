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
f = './data/coco2.png'
z = e(f)
#e.debug_rhip(z)
#e.debug_rknee(z)
#e.debug_rotation(z)
data = e.kinematicTree(z)
print(data)

np.save('k_tree.npy', data)
