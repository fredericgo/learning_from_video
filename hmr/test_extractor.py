from absl import flags
import src.config
import sys
from src.SkeletonExtractor import SkeletonExtractor

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1

e = SkeletonExtractor(config)
f = './data/coco1.png'
z = e(f)
#e.debug_rhip(z)
e.kinematicTree(z)
