from absl import flags
import src.config
import sys
from src.SkeletonExtractor import SkeletonExtractor

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1

extract = SkeletonExtractor(config)
f = './data/cam5_frame000153.jpg'
extract(f)
