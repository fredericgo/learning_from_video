from absl import flags
import os
import src.config
import sys
from src.SkeletonExtractor import SkeletonExtractor
import numpy as np

config = flags.FLAGS
config(sys.argv)
config.load_path = src.config.PRETRAINED_MODEL
config.batch_size = 1

e = SkeletonExtractor(config)

img_dir = "data/youtube/1/"
onlyfiles = [f for f in os.listdir(img_dir)
             if os.path.isfile(os.path.join(img_dir, f))]
onlyfiles = sorted(onlyfiles, 
                   key=lambda f: int(f.rsplit('.')[0].split('_')[-1]))

print("Extracting from directory: {}".format(img_dir))
N = len(onlyfiles)
X = np.zeros((N,24))

for i, file in enumerate(onlyfiles):
    print("File: {}".format(file))
    img_path = os.path.join(img_dir, file)
    z = e(img_path)
    data = e.kinematicTree(z)
    X[i] = data

np.save('results/k_trees.npy', data)
#np.save('j3d.npy', z)
